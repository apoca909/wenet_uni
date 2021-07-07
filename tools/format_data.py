import os

def getname(fullpath, omit='.wav'):
    name = os.path.basename(fullpath)
    name = name.strip(omit)
    return name

def tokens2ids(text, vocab, unkid=1):
    ids = [vocab.get(c, unkid) for c in text]
    return ids

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

class format_data():
    def __init__(self, ptext, pscp, pvocab, pout, ptxt, scp):
        self.ptext = ptext
        self.pscp = pscp
        self.pvocab = pvocab
        self.pout=pout
        self.ptxt=ptxt
        self.scp = scp

    def init_vocab(self,):
        self.token_ids = { }
        for line in open(self.pvocab, 'r', encoding='utf-8'):
            seq = line.rfind(' ')
            if seq > 0:
                self.token_ids[line[0:seq]] = line[seq+1:].strip()
            else:
                self.token_ids[line.strip()] = len(self.token_ids)

        self.vocab_size = len(self.token_ids)

    def build_vocab(self,):
        self.token_ids = {'<blank>':0, '<unk>': 1}
        tokens = set()
        for line in open(self.ptext, 'r', encoding='utf-8'):
            line_text = line.split('\t')[1].strip()
            #line_text = line_text.replace(' ', '|')
            text_chs = set(line_text)
            tokens.update(text_chs)

        for i, c in enumerate(tokens):
            self.token_ids[c] = len(self.token_ids)

        self.token_ids['<sos/eos>'] = len(self.token_ids)
        self.vocab_size = len(self.token_ids)

        with open(self.pvocab, 'w', encoding='utf-8') as fw:
            for k, v in self.token_ids.items():
                fw.write(f'{k} {v}\n')

    def build_wedata(self,):
        text_lines = [line.strip() for line in open(self.ptext, 'r', encoding='utf-8')]
        tsv_lines = [line.strip() for line in open(self.pscp, 'r', encoding='utf-8')]

        fw = open(self.pout, 'w', encoding='utf-8')
        fw_txt = open(self.ptxt, 'w', encoding='utf-8')
        fw_scp = open(self.scp, 'w', encoding='utf-8')
        oov = {}
        duration_all = 0
        max_len = 0
        fw.write('\n')
        for i in range(len(text_lines)):
            if text_lines[i].strip() == "":
                continue
            if tsv_lines[i].strip() == "":
                continue

            if len(text_lines[i].strip().split('\t')) != 2:
                continue
            wav_path, text_tok = text_lines[i].strip().split('\t')
            if len(tsv_lines[i].strip().split('\t')) != 3:
                continue

            _, frames, sr = tsv_lines[i].strip().split('\t')

            duration = eval(frames)/eval(sr)

            if duration > 20 : continue

            if duration < 0.5: continue

            name = getname(wav_path)

            #text = text_tok.replace(' ', '')
            text = get_hex(text_tok, cws_tag='|', char_tag=' ')
            for c in text:
                if c not in self.token_ids:
                    oov[c] = oov.get(c, 0) + 1

            ids = tokens2ids(text.split(), self.token_ids)
            if len(ids) > max_len:
                max_len  = len(ids)
            if len(ids) > 120 : continue
            
            ids_str = ' '.join([str(id) for id in ids])
            token_shape = f'{len(ids)},{self.vocab_size}'
            duration_all += duration
            item = f'utt:{name}\tfeat:{wav_path}\tfeat_shape:{duration:.3}\ttext:{text_tok}\ttoken:{text}\ttokenid:{ids_str}\ttoken_shape:{token_shape}\n'
            fw.write(item)
            fw_txt.write(f'{name} {text}\n')
            fw_scp.write(f'{name} {wav_path}\n')

            if i % 1000 == 0:
                fw.flush()
                fw_txt.flush()
                fw_scp.flush()
        print(f' duration : {self.ptext} {duration_all/3600} maxlen {max_len}')
        if not os.path.exists('./oov.txt'):
            with open('./oov.txt', 'w', encoding='utf-8') as fw_oov:
                oov_ = sorted(oov.items(), key=lambda  x: x[1], reverse=True)
                for k, v in oov_:
                    fw_oov.write(f'{k}\t{v}\n')
        fw.flush()
        fw.close()
        fw_txt.flush()
        fw_txt.close()
        fw_scp.flush()
        fw_scp.close()

def build_wedata():

    train_data = ['biaozhu_data',  'ch_x_en_data', 'free_talk_100h', 'Mandarin300h', 'nature_talk', 'qihoo_201408', 'azero_ripple',
                  #'AISHELL-ASR0007','APY161101014_G_159hours_kouyin_100hours', 'APY161101014_R_754hours_kouyin_100hours', 'APY161101014_R_754hours_kouyin_141hours', '703person_natural_communicate_400h_cut',
                  ]
    dev_data = ['dangbei_20210516', 'meet1', 'meet2', 'valid']

    for data in train_data:
        wedata_train = format_data(ptext=f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/{data}.script',
                                   pscp=f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/{data}.tsv',
                                   pvocab='/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/data/local/dict/units.txt',
                                   pout=f'/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/train/{data}.udata',
                                   ptxt=f'/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/train/{data}',
                                   scp=f'/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/train/{data}.scp')
        wedata_train.init_vocab()
        wedata_train.build_wedata()
    os.system('cat /audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/train/*.udata > /audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/train/train.udata')
    for data in dev_data:
        wedata_dev = format_data(ptext=f'/audiodata6/zhaoang/workspace/wav_scp/scripts/dev/{data}.script',
                                 pscp=f'/audiodata6/zhaoang/workspace/wav_scp/scripts/dev/{data}.tsv',
                                 pvocab='/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/data/local/dict/units.txt',
                                 pout=f'/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/dev/{data}.udata',
                                 ptxt=f'/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/dev/{data}',
                                 scp=f'/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/dev/{data}.scp')
        wedata_dev.init_vocab()
        wedata_dev.build_wedata()
    os.system('cat /audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/dev/*.udata > /audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/dev/dev.udata')


def decode_unichar(fin, fout):
    with open(fout, 'w', encoding='utf-8') as fw:
        for i, line in enumerate(open(fin, 'r', encoding='utf-8')):
            if line.strip():
                pos = line.find('\t')
                if pos == -1:
                    pos = line.find(' ')
                name, text_uni = line[0:pos], line[pos+1:].strip()
                #print(fin, line)
                try:
                    text_ch = get_hex(line=text_uni, mode='dec', cws_tag='|', char_tag=' ')
                except Exception as e:
                    text_ch = text_uni
                fw.write(f'{name} {text_ch}\n')

if __name__ == '__main__':
    # text_tok = '播放 萨 顶顶 的 不 染'
    # text = get_hex(text_tok, cws_tag='|', char_tag=' ')
    # print(text)
    #build_wedata()
    #build_dangbei()
    #print(get_hex(line='E5819CE6ADA2|', mode='dec', cws_tag='|', char_tag=' '))
    print(get_hex(line="e4bda0|e68a8a|e68891e79a84|e68891e79a84|e68891e79a84|e68891e79a84|e69c8be58f8b|e5958a|" , mode='dec', cws_tag='|', char_tag=' '))
    #decode_unichar('/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/dev/text2', '/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/raw_wav/dev/text2char')
    #decode_unichar('/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/exp/conformer/test_kenlm_decoder/text_pred', '/audiodata6/zhaoang/workspace/wenet/examples/aishell/s0/exp/conformer/test_kenlm_decoder/text_pred2char')





