#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import argparse
import h5py
import numpy as np
import os
from scipy.signal import medfilt

parser = argparse.ArgumentParser(description='make rttm from decoded result')
parser.add_argument('file_list_hdf5')
parser.add_argument('out_rttm_file')
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--frame_shift', default=256, type=int)
parser.add_argument('--subsampling', default=1, type=int)
parser.add_argument('--median', default=1, type=int)
parser.add_argument('--sampling_rate', default=16000, type=int)
parser.add_argument('--chunksize', default=2000, type=int)
args = parser.parse_args()

filepaths = [line.strip() for line in open(args.file_list_hdf5)]
filepaths.sort()

with open(args.out_rttm_file, 'w') as wf:
    for filepath in filepaths:
        session, _ = os.path.splitext(os.path.basename(filepath))
        data = h5py.File(filepath, 'r')
        a = np.where(data['T_hat'][:] > args.threshold, 1, 0)
        # adust the overlap part for loooooooooong audio
        # default overlap is 100 frames
        if a.shape[0] > args.chunksize:  # it means there should be overlap happening
             mask = np.ones(a.shape[0],)
             assert a.shape[1] == 2, 'Only support 2 speakers temporarily'
             for sep_point in range(args.chunksize, a.shape[0], args.chunksize):
                 # compare left 100 and right 100 frames
                 assert sep_point+100 < a.shape[0], f'next chunk length as least 100 frames, but with {sep_point+100} > {a.shape[0]}'
                 lfs = a[sep_point-100:sep_point]
                 rfs = a[sep_point:sep_point+100]
                 sim_spk_1_1 = (lfs[:,0] == rfs[:,0]).sum()
                 sim_spk_1_2 = (lfs[:,0] == rfs[:,1]).sum()
                 if sim_spk_1_1 < sim_spk_1_2: # should reverse the id assignment
                     print(f'Reverse since {sim_spk_1_1} < {sim_spk_1_2}')
                     a[sep_point:min(sep_point+args.chunksize, a.shape[0])] = a[sep_point:min(sep_point+args.chunksize, a.shape[0]),::-1] # reverse the colunms
                 mask[sep_point:sep_point+100] = 0 # get rid of duplicated parts, we remove the latter
             mask = mask == 1
             a = a[mask]

        if args.median > 1:
            a = medfilt(a, (args.median, 1))
        for spkid, frames in enumerate(a.T):
            frames = np.pad(frames, (1, 1), 'constant')
            changes, = np.where(np.diff(frames, axis=0) != 0) # np.diff：当前帧-上一帧，如果不=0，说明两帧之间不一样咯
            fmt = "SPEAKER {:s} 1 {:.2f} {:.2f} <NA> <NA> {:d} <NA> <NA>"
            for s, e in zip(changes[::2], changes[1::2]):
                print(fmt.format(
                      session,
                      s * args.frame_shift * args.subsampling / args.sampling_rate,  # 把帧数据转换成秒，这里的参数一定要对齐
                      (e - s) * args.frame_shift * args.subsampling / args.sampling_rate,
                        int(spkid)), file=wf)

# session + "_" + str(spkid)