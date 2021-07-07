import argparse
import os
import soundfile


def scan_one_scp(fscp):
    not_exists_paths = []
    tsvs = []
    for i, line in enumerate(open(fscp, 'r', encoding='utf-8')):
        line = line.strip()
        try:
            path, _ = line.split('\t')
            if not os.path.exists(path):
                not_exists_paths.append(path)
                continue
            info = soundfile.info(path)
            frames, sr = info.frames, info.samplerate
            if sr != 16000:
                print(f'{path}, {sr}')
            tsvs.append((path, frames, sr))
        except Exception as e:
            print(fscp, i , line)

    return tsvs, not_exists_paths
##123
def gen_tsv():
    dir = '/audiodata6/zhaoang/workspace/wav_scp/scripts/dev/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--scps", default=[
        f"{dir}dangbei_20210516.script",
        f"{dir}meet1.script",
        f"{dir}meet2.script",
        f"{dir}valid.script",
    ], required=False, help="Path to transcript file")

    args = parser.parse_args()

    for f in args.scps:
        tsvs, not_exists_paths = scan_one_scp(f)
        duration = sum([t[1] / t[2] for t in tsvs]) / 3600.

        tsvs = [f'{t[0]}\t{t[1]}\t{t[2]}' for t in tsvs]
        f_tsv = f.replace('.script', '.tsv')
        f_lost = f.replace('.script', '.lost')
        with open(f_tsv, 'w', encoding='utf-8') as ff_tsv, open(f_lost, 'w', encoding='utf-8') as ff_lost:
            ff_lost.write('\n'.join(not_exists_paths))
            ff_lost.write(str(duration))

            ff_tsv.write('\n'.join(tsvs))
#gen_tsv()

def merge():
    dd = 'D:/data/ASR_DATA/weekly_data/azero_asr_ripple_test'
    for f in os.listdir(dd):
        if f.endswith('.text'):
            text = os.path.join(dd, f)
            script = os.path.join(dd, f.replace('.text', '.script'))

            text_lines = [line.strip() for line in open(text, 'r', encoding='utf-8')]
            script_lines = [line.split('\t')[0] for line in open(script, 'r', encoding='utf-8')]

            with open(script, 'w', encoding='utf-8') as fw:
                for i, line in enumerate(script_lines):
                    fw.write(line + '\t' + text_lines[i] + '\n')

def get_text():
    dd = 'D:/data/ASR_DATA/weekly_data/azero_asr_ripple_test'
    targets = ['azero_ripple.script']
    for f in os.listdir(dd):
        if f.endswith('.script') and f in targets:
            script = os.path.join(dd, f)
            script_lines = [''.join(line.strip().split('\t')[1:]).replace(' ','') for line in open(script, 'r', encoding='utf-8')]

            with open(script.replace('.script','.text'), 'w', encoding='utf-8') as fw:
                for i, line in enumerate(script_lines):
                    fw.write(line + '\n')

def recheck():
    ptext = f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/azero_ripple.script'
    pscp = f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/azero_ripple.tsv'

    ptext_lines = [line for line in open(ptext, 'r', encoding='utf-8')]
    pscp_lines = {line.split('\t')[0]:line for line in open(pscp, 'r', encoding='utf-8')}

    ptext_lines_f = []
    pscp_lines_f = []
    for line in ptext_lines:
        path = line.split('\t')[0]
        if path in pscp_lines and os.path.exists(path):
            ptext_lines_f.append(line)
            pscp_lines_f.append(pscp_lines[path])

    with open(f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/azero_ripple.script.1', 'w', encoding='utf-8') as fw1, \
            open(f'/audiodata6/zhaoang/workspace/wav_scp/scripts/train/azero_ripple.tsv.1', 'w',
                 encoding='utf-8') as fw2:
        fw1.write(''.join(ptext_lines_f))
        fw2.write(''.join(pscp_lines_f))




#merge()