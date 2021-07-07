import paramiko
import os
from playsound import playsound


def ftpconnect(host, username, password):
        sf  = paramiko.Transport((host, 22))  # 设置变量
        sf.connect(username=username, password=password) # 连接FTP服务器
        sftp = paramiko.SFTPClient.from_transport(sf)
        return sf, sftp

def downloadfile(sftp, remotepath, localdir):
        fname = os.path.basename(remotepath)
        localpath = os.path.join(localdir, fname) # 定义文件保存路径
        if os.path.exists(localpath):
            return 1
        sftp.get(remotepath, localpath)
        return 0

def merge_scp_text(scpfile, text_file, tar_file):
    scps = {}
    texts = {}
    for line in open(scpfile, 'r', encoding='utf-8'):
        fname, trans = line.strip().split('\t')
        scps[fname] = trans

    fw = open(tar_file, 'w', encoding='utf-8')
    for i, line in enumerate(open(text_file, 'r', encoding='utf-8')):
        if len(line.strip().split('\t')) != 2:
            continue
        fname, trans = line.strip().split('\t')
        texts[fname] = trans
        full_path = scps.get(fname)
        if full_path is not None:
            fw.write(f'{full_path}\t{trans}\n')

def crawl():
    sf, ftp = ftpconnect('172.16.10.23', "l_zhaoang", 'zhaoang123')
    wavdir = '/audiodata/audio_public/asr/biaozhu_data/wavs/'

    names = ['00-06-33-277_a2f5a314cbf20553e2dd0e6423d1435b_ALL',]

    paths = [f'{wavdir}{name}.wav' for name in names]
    localdir = 'D:/abc'
    for i, fname in enumerate(paths):
        if downloadfile(ftp, fname, localdir) == 1:
            pass
        playsound(os.path.join(localdir, os.path.basename(fname)))

if __name__ == "__main__":
    #merge_scp_text('D:/data/ASR_DATA/yinxiang/trans/total.scp', 'D:/data/ASR_DATA/yinxiang/trans/total.text', 'D:/data/ASR_DATA/yinxiang/trans/test.script')
    crawl()