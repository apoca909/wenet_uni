# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
#from contextlib import nullcontext
# if your python version < 3.7 use the below one
from contextlib import suppress as nullcontext
import torch
from torch.nn.utils import clip_grad_norm_
import pynvml
pynvml.nvmlInit()

import numpy as np
import torch

def check_tensor(vector, name=None):
    try:
        if isinstance(vector, torch.Tensor):
            if not np.any(np.isnan(vector.cpu().detach().numpy())):
                return True
            return False
        elif isinstance(vector, np.ndarray):
            if not np.any(np.isnan(vector)):
                return True
            return False
        elif isinstance(vector, list):
            vector = np.asarray(vector, dtype=np.float32)
            if not np.any(np.isnan(vector)):
                return True
            return False
    except Exception as ex:
        logging.error("name is [{}] value is {}".format(name, vector))
        return False

class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        logging.info('using accumulate grad, new batch size is {} times'
                     'larger than before'.format(accum_grad))
        handle = pynvml.nvmlDeviceGetHandleByIndex(rank)

        if use_amp:
            assert scaler is not None
        num_seen_utts = 0

        num_total_batch = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            key, feats, target, feats_lengths, target_lengths = batch
            #check_tensor(feats)
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            num_utts = target_lengths.size(0)
            if num_utts == 0:
                continue
            context = None
            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            if is_distributed and batch_idx % accum_grad != 0:
                context = model.no_sync
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            else:
                context = nullcontext
            with context():
                # autocast context
                # The more details about amp can be found in
                # https://pytorch.org/docs/stable/notes/amp_examples.html
                try:
                    with torch.cuda.amp.autocast(scaler is not None):
                        loss, loss_att, loss_ctc = model(feats, feats_lengths, target, target_lengths)
                        loss = loss / accum_grad
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                except Exception as e:
                    print(e, feats_lengths)
                    continue

            num_seen_utts += num_utts
            if batch_idx % accum_grad == 0:
                if rank == 0 and writer is not None:
                    writer.add_scalar('train_loss', loss, self.step)
                # Use mixed precision training
                if use_amp:
                    scaler.unscale_(optimizer)
                    grad_norm = clip_grad_norm_(model.parameters(), clip)
                    # Must invoke scaler.update() if unscale_() is used in the
                    # iteration to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # We don't check grad here since that if the gradient has
                    # inf/nan values, scaler.step will skip optimizer.step().
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = clip_grad_norm_(model.parameters(), clip)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                self.step += 1
            
            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                    batch_idx, num_total_batch,
                    loss.item() * accum_grad)
                if loss_att is not None:
                    log_str += 'loss_att {:.6f} '.format(loss_att.item())
                if loss_ctc is not None:
                    log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                logging.debug(log_str)
            
            if batch_idx % 12 == 0:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                usage = meminfo.used / meminfo.total
                if usage > 0.55:
                    torch.cuda.empty_cache()
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    usage_ec = meminfo.used / meminfo.total
                    logging.info(f'gpu {rank} memory usage {usage:.4} -> {usage_ec:.4}')

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = args.get('log_interval', 10)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        num_total_batch = len(data_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                loss, loss_att, loss_ctc = model(feats, feats_lengths, target,
                                                 target_lengths)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        batch_idx, num_total_batch, loss.item())
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    logging.debug(log_str)

        return total_loss, num_seen_utts
