import os
import shutil
from collections import defaultdict

def float2time(value):
  hr = int(value / 3600)
  value = value % 3600
  m = int(value / 60)
  value = value % 60
  sec = value
  return '{}:{}:{:.3}'.format(str(hr).rjust(2, '0'), str(m).rjust(2, '0'), sec)

exp = ['final_4'] #['exp_1', 'exp_2', 'exp_3', 'exp_4']

for e in exp:
  results = f'exp/{e}.rttm'
  speakers = defaultdict(list)
  cnt = 1
  with open(results, 'r') as fr:
    lines = fr.readlines()
    lines = sorted(lines, key=lambda x:float(x.split()[3]))
    for line in lines:
      print(line)
      _, _, _, st, dur, _, _, sp, _ = line.split()
      st = float(st)
      st_ = float2time(st)
      #print(float2time(st))
      dur = float(dur)
      dur_ = float2time(dur)
      if not os.path.exists(os.path.join(e)):
        #shutil.rmtree(os.path.join(e, sp))
        os.mkdir(e)
      cnt += 1
      print(f'ffmpeg -i sample/{e}.wav -ss {st_} -t {dur_} {e}/{cnt}_{sp}.wav')
      os.system(f'ffmpeg -i sample/{e}.wav -ss {st_} -t {dur_} {e}/{cnt}_{sp}.wav')
      
      
