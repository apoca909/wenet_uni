# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
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
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.transformer.LexiconTrie import Vocab

from functools import wraps
import time

def fun_run_time(func):
    @wraps(func)
    def inner(*args, **kwargs):
        s_time = time.time()
        ret = func(*args, **kwargs)
        e_time = time.time()
        print("{} cost {} s".format(func.__name__, e_time - s_time))
        return ret
    return inner

@fun_run_time
def run_recognize():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=False,default='exp/conformer/train.yaml', help='config file')
    parser.add_argument('--test_data', default='raw_wav/dev/format3.data',required=False, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', default='exp/conformer/avg_2.pt',required=False, help='checkpoint model')
    parser.add_argument('--dict', required=False, help='dict file')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', default='exp/conformer3/test_attention_rescoring/text_pred', help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring', 'kenlm_decoder'],
                        default='attention_rescoring',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.5,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')

    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')

    parser.add_argument('--tgt_dict', default='data/dict/unicode_vocab.txt')
    parser.add_argument('--pkenlm_model', default='/audiodata6/zhaoang/workspace/wav_scp/kenlm_unicode/lm.bin')
    parser.add_argument('--plexicon', default='/audiodata6/zhaoang/workspace/wav_scp/kenlm_unicode/lexicon.txt')

    args = parser.parse_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal('decoding mode {} must be running with batch_size == 1'.format(args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    raw_wav = configs['raw_wav']
    # Init dataset and data loader
    # Init dataset and data loader
    test_collate_conf = copy.deepcopy(configs['collate_conf'])
    test_collate_conf['spec_aug'] = False
    test_collate_conf['spec_sub'] = False
    test_collate_conf['feature_dither'] = False
    test_collate_conf['speed_perturb'] = False
    if raw_wav:
        test_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
    test_collate_func = CollateFunc(**test_collate_conf,
                                    raw_wav=raw_wav)
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['batch_type'] = 'static'
    dataset_conf['sort'] = False
    test_dataset = AudioDataset(args.test_data, **dataset_conf, raw_wav=raw_wav)
    test_data_loader = DataLoader(test_dataset,
                                  collate_fn=test_collate_func,
                                  shuffle=False,
                                  batch_size=1,
                                  num_workers=0)

    # Init asr model from configs
    model = init_asr_model(configs)

    # Load dict

    tgt_dict = Vocab(dict_path=args.dict)
    eos = tgt_dict.eos_idx

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()
    #lm_decoder = W2lKenLMDecoder(args.pkenlm_model,args.plexicon, tgt_dict)
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            if args.mode == 'attention':
                hyps = model.recognize(
                    feats,
                    feats_lengths,
                    beam_size=args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp.tolist() for hyp in hyps]
            elif args.mode == 'ctc_greedy_search':
                hyps = model.ctc_greedy_search(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
            # ctc_prefix_beam_search and attention_rescoring only return one
            # result in List[int], change it to List[List[int]] for compatible
            # with other batch decoding mode
            elif args.mode == 'ctc_prefix_beam_search':
                assert (feats.size(0) == 1)
                hyp = model.ctc_prefix_beam_search(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp]
            elif args.mode == 'attention_rescoring':
                assert (feats.size(0) == 1)
                hyp = model.attention_rescoring(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming)

                hyps = [hyp]

            for i, key in enumerate(keys):
                #logging.info(f"{tgt_dict.dict_size}{hyps[i]}")
                content = ''.join(tgt_dict.ids2str(hyps[i]))
                logging.info('{} {}'.format(key, content))
                fout.write('{}\t{}\n'.format(key, content))
if __name__ == '__main__':
    run_recognize()

