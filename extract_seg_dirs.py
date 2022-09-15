import os
import shutil
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('rttm', help='infer results')
parser.add_argument('wavscp', help='just the test wavscp')
parser.add_argument('outdir', help='just the test wavscp')

args = parser.parse_args()

rttm = args.rttm
wavscp = args.wavscp
outdir = args.outdir

if not os.path.exists(outdir):
    os.mkdir(outdir)

# 0. useful function
def float2time(value):
  hr = int(value / 3600)
  value = value % 3600
  m = int(value / 60)
  value = value % 60
  sec = value
  return '{}:{}:{:.3}'.format(str(hr).rjust(2, '0'), str(m).rjust(2, '0'), sec)

# 1. read all the rttm info
assert os.path.exists(rttm), 'rttm not exists!! Pls check again'
rlts = defaultdict(list)
with open(rttm, 'r') as fr:
    for line in fr.readlines():
        recid = line.strip().split()[1]
        rlts[recid].append(line)

#2. read recid -> wavs
recid2wav = {}
with open(wavscp, 'r') as fr:
    for line in fr.readlines():
        recid, wav_path = line.strip().split()
        recid2wav[recid] = wav_path
        
# 3. processing each file
for recid in rlts:
    wav_path = recid2wav[recid]
    filename = os.path.basename(wav_path).split('.')[0]
    lines = rlts[recid]

    speakers = defaultdict(list)
    cnt = 0
    lines = sorted(lines, key=lambda x:float(x.split()[3]))
    e = os.path.join(outdir, filename)
    for line in lines:
        _, _, _, st, dur, _, _, sp, _ = line.split()
        st = float(st)
        st_ = float2time(st)
        #print(float2time(st))
        dur = float(dur)
        dur_ = float2time(dur)
        if not os.path.exists(e):
            #shutil.rmtree(os.path.join(e, sp))
            os.mkdir(e)
        cnt += 1
        print(f'ffmpeg -i {wav_path} -ss {st_} -t {dur_} {e}/{cnt}_{sp}.wav')
        os.system(f'ffmpeg -i {wav_path} -ss {st_} -t {dur_} {e}/{cnt}_{sp}.wav')
      
      
