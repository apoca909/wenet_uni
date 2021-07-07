import os
import numpy as np
import soundfile
import scipy.io.wavfile as wav
import textgrid

def get_textgrid_wav_pair(sdir, end='.TextGrid'):
    files = []
    for file in os.listdir(sdir):
        file = os.path.join(sdir, file)
        if file.endswith(end):
            wav = file.replace(end, '.wav')
            if not os.path.exists(wav):
                print(f'{wav} not exists')
                continue
            files.append((file, wav))
    return files

def get_text_grid(text_grid_file):
    tg = textgrid.TextGrid.fromFile(text_grid_file)
    return tg

def cut_one_wav(tgs, wavpath, tdir):
    sig, rate = soundfile.read(wavpath)
    sig = sig * 32767
    sig = sig.astype(np.int16)

    texts = []
    namebase = os.path.basename(wavpath).replace('.wav', '')
    for tg in tgs:
        for interval in tg:
            start = interval.minTime
            end = interval.maxTime
            text = interval.mark
            if text is None or text is "":
                continue
            text = text.strip('\"')
            start_index = int(start * rate)
            end_index = int(end * rate)
            piece = sig[start_index:end_index]
            piecename = namebase + '_' + str(round(start, 2)) + '_' + str(round(end, 2))
            wav.write(os.path.join(tdir, piecename + '.wav'), rate, piece)
            texts.append(piecename + '\t' + text)
    return texts

def cut_wav_pipline(sdir):
    sdir = sdir.strip('/')
    tdir = sdir + '_cut'
    fwtext = sdir + '_text'
    if not os.path.exists(tdir):
        os.mkdir(tdir)

    files = get_textgrid_wav_pair(sdir)
    texts = []
    for textgridfile, wavfile in files:
        tgs = get_text_grid(textgridfile)
        one_wav_text = cut_one_wav(tgs, wavfile, tdir)
        texts.extend(one_wav_text)

    with open(fwtext, 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(texts))

cut_wav_pipline(sdir='D:/data/ASR_DATA/szp')






