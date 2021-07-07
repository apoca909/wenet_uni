from os.path import abspath
import os
import argparse

def get_hex(line, mode='enc', cws_tag = ' ', char_tag = ''):
    if mode == 'enc':
        x = bytes(line.encode('utf-8'))
        line = ''
        for i in x:
            if i == 0x20:
                line += cws_tag + char_tag
            else:
                line += f'{i:x}' + char_tag
        line += cws_tag
    elif mode == 'dec':
        line = line.replace(' ', '')
        words = []
        for uword in line.split('|'):
            try:
                word = bytes.fromhex(uword).decode('utf-8')
            except Exception as e:
                word = "<unk>"
            words.append(word)
        line = ' '.join(words)
    return line.upper()

def step2(strs):
    strs_n = ''
    for i in range(int(len(strs) / 2)):
        strs_n += strs[i * 2:i * 2 + 2] + ' '
    return strs_n.strip()


def cat_traindata(train_text_path, files):
    fw_text_lm = open(train_text_path, 'w', encoding='utf-8')

    lines = []
    for f in files:
        fr = open(f, 'r', encoding='utf-8')
        for line in fr:
            if len(line.strip().split('\t')) == 2:
                line = line.strip().split('\t')[1]
            line = line.upper()
            line = get_hex(line, cws_tag='| ', char_tag='')
            lines.append(line)

            if (len(lines) + 1) % 10000 == 0:
                fw_text_lm.write('\n'.join(lines) + '\n')
                lines = []

    fw_text_lm.write('\n'.join(lines) + '\n')

def main_kenlm():
    parser = argparse.ArgumentParser()

    parser.add_argument("--transcript_file",
                        default=[f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/biaozhu_data.script',
                                 f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/azero_ripple.script',
                                 f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/ch_x_en_data.script',
                                 f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/free_talk_100h.script',
                                 f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/Mandarin300h.script',
                                 f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/nature_talk.script',
                                 f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/qihoo_201408.script',
                                 f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/AISHELL-ASR0007.script',
                                 f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/703person_natural_communicate_400h_cut.script',
                                 ],
                        type=list, required=False,
                        help="Path to text file")

    parser.add_argument("--ngram", default=3, type=int,
                        required=False, help="Ngram")


    parser.add_argument("--in_path", default='/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/data/local/dict', type=str,
                        required=False, help="Output path for storing model")

    parser.add_argument("--output_path_lm",
                        default='/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/data/local/lm', type=str,
                        required=False, help="Output path for storing model")

    args = parser.parse_args()


    #lexicon_path = os.path.join(args.in_path, 'lexicon.txt')
    words_path = os.path.join(args.in_path, 'words.txt')
    train_text_path = os.path.join(args.output_path_lm, 'world_lm_data.train')
    model_arpa = os.path.join(args.output_path_lm, 'lm.arpa')

    lexions = [ ]
    for line in open(words_path, 'r', encoding='utf-8'):
        word = line.strip()
        word_u = get_hex(word, cws_tag='|', char_tag='')
        word_split = get_hex(word, cws_tag='|', char_tag=' ')
        lexion = word_u + '\t' + word_split
        lexions.append(lexion)

    lexicon_path = os.path.join(args.in_path, 'lexicon.txt')
    print('\n'.join(lexions), file=open(lexicon_path, 'w', encoding='utf-8'))

    words = [get_hex(line.strip(), cws_tag = '|', char_tag='') for line in open(words_path, 'r', encoding='utf-8')]
    uwords_path = os.path.join(args.in_path, 'words.txt.u')
    print('\n'.join(words), file=open(uwords_path, 'w', encoding='utf-8'))

    cmd = f" /root/kenlm/build/bin/lmplz -T /tmp -S 4G --prune 0 5 5 --discount_fallback -o {args.ngram} " \
          f" --limit_vocab_file {uwords_path} trie < {train_text_path} > {model_arpa}"
    #语言模型
    #cat_traindata(train_text_path, args.transcript_file)
    print(cmd)
    #os.system(cmd)
    #构图
    cmd2 = 'cd examples/aishell/s0 && tools/fst/compile_lexicon_token_fst.sh data/local/dict data/local/tmp data/local/lang' \
           ' && tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1'

    os.system(cmd2)
#bash run_yx.sh --stage 6 --stop-stage 6

main_kenlm()
print(get_hex('我的 一天', cws_tag = '| ', char_tag = ''))