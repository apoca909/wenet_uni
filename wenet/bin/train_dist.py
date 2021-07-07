# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.append("../../")

from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint, save_checkpoint
from wenet.utils.executor import Executor
from wenet.utils.scheduler import WarmupLR, get_linear_schedule_with_warmup


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', default='/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/conf/train_conformer.yaml', help='config file')
    parser.add_argument('--train_data', default='/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/train/train.udata', help='train data file')
    parser.add_argument('--cv_data', default='/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/dev/dev.udata', help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='backend')
    parser.add_argument('--model_dir', default='/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/exp/conformer_unicode', help='save model dir')
    parser.add_argument('--checkpoint', default=None, help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='local rank/gpu')
    parser.add_argument('--num_workers',
                        default=12,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        default=True,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Set random seed
    torch.manual_seed(777)
    logging.info(args)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    world_size = torch.cuda.device_count()
    distributed = world_size > 1
    local_rank = args.local_rank
    raw_wav = configs['raw_wav']


    train_collate_func = CollateFunc(**configs['collate_conf'], raw_wav=raw_wav)

    cv_collate_conf = copy.deepcopy(configs['collate_conf'])
    # no augmenation on cv set
    cv_collate_conf['spec_aug'] = False
    cv_collate_conf['spec_sub'] = False
    if raw_wav:
        cv_collate_conf['feature_dither'] = 0.0
        cv_collate_conf['speed_perturb'] = False
        cv_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
    cv_collate_func = CollateFunc(**cv_collate_conf, raw_wav=raw_wav)

    dataset_conf = configs.get('dataset_conf', {})
    train_dataset = AudioDataset(args.train_data,
                                 **dataset_conf,
                                 raw_wav=raw_wav)
    cv_dataset = AudioDataset(args.cv_data, **dataset_conf, raw_wav=raw_wav)

    if distributed:
        dist.init_process_group(backend=args.backend,
                                init_method=None, #'env://'
                                world_size=world_size,
                                rank=local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        logging.info('training on {} gpus, this gpu {}'.format(world_size, local_rank))

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
        cv_sampler = torch.utils.data.distributed.DistributedSampler(
            cv_dataset, shuffle=False)
    else:
        train_sampler = None
        cv_sampler = None
        device = torch.device('cuda' if args.gpu > -1 else 'cpu')

    train_data_loader = DataLoader(train_dataset,
                                   collate_fn=train_collate_func,
                                   sampler=train_sampler,
                                   shuffle=(train_sampler is None),
                                   pin_memory=args.pin_memory,
                                   batch_size=1,
                                   num_workers=args.num_workers)
    cv_data_loader = DataLoader(cv_dataset,
                                collate_fn=cv_collate_func,
                                sampler=cv_sampler,
                                shuffle=False,
                                batch_size=1,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers)

    if raw_wav:
        input_dim = configs['collate_conf']['feature_extraction_conf'][
            'mel_bins']
    else:
        input_dim = train_dataset.input_dim
    vocab_size = train_dataset.output_dim

    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = raw_wav
    if local_rank == 0:
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Init asr model from configs
    model = init_asr_model(configs)
    #print(model)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info('the number of model params: {}'.format(num_params))

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    script_model = torch.jit.script(model)
    script_model.save(os.path.join(args.model_dir, 'init.zip'))
    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    writer = None
    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if distributed:
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=configs['scheduler_conf']['warmup_steps'], num_training_steps=1000000)
    final_epoch = None
    configs['rank'] = local_rank
    configs['is_distributed'] = distributed
    configs['use_amp'] = args.use_amp
    if start_epoch == 0 and local_rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # Start training loop
    executor.step = step
    #scheduler.set_step(step)

    # used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_data_loader, device,
                       writer, configs, scaler)
        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device, configs)
        if world_size > 1:
            # all_reduce expected a sequence parameter, so we use [num_seen_utts].
            num_seen_utts = torch.Tensor([num_seen_utts]).to(device)
            # the default operator in all_reduce function is sum.
            dist.all_reduce(num_seen_utts)
            total_loss = torch.Tensor([total_loss]).to(device)
            dist.all_reduce(total_loss)
            cv_loss = total_loss[0] / num_seen_utts[0]
            cv_loss = cv_loss.item()
        else:
            cv_loss = total_loss / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        if local_rank == 0:
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': cv_loss,
                    'step': executor.step
                })
            writer.add_scalars('epoch', {'cv_loss': cv_loss, 'lr': lr}, epoch)
        final_epoch = epoch

    if final_epoch is not None and local_rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)


#python train_lm.py
#python format_data.py
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=8 train_dist.py --use_amp