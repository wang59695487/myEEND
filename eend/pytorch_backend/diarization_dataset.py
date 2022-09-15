#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import torch
import numpy as np
from eend import kaldi_data
from eend import feature


def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size=2000, step=2000,
        use_last_samples=False,
        label_delay=0,
        subsampling=1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


def my_collate(batch):
    data, target = list(zip(*batch))
    return [data, target]


class KaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,
            chunk_size=2000,
            context_size=0,
            frame_size=1024,
            frame_shift=256,
            subsampling=1,
            rate=16000,
            input_transform=None,
            use_last_samples=False,
            label_delay=0,
            n_speakers=None,
            ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling 
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.chunk_indices = []
        self.label_delay = label_delay

        self.sr = rate
        self.data = kaldi_data.KaldiData(self.data_dir) # not support well yet

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs: # 这里的rec是指speaker，广义的speaker，即speaker-chapter-sentence，但分类是还是按speaker来的
            # 这里没有segments文件的使用，说明每个文件已经拆到较小的粒度了，不需要额外再做句子分割了
            # 修正：segments文件在feature.get_labededSTFT里用到了……
            # 我理解segments中的格式：
            #                       speaker-conversationID-segID start_time end_time
            #                       {'utt':'1-1001-1', 'st':xx, 'et':xx},
            #                       2-1001-2 st ed
            #                       1-1001-3 st ed
            #                       ...
            data_len = int(self.data.reco2dur[rec] * rate / frame_shift) # logmel特征的数量 
            data_len = int(data_len / self.subsampling) # 这“句”音频可提取的输入进网络的帧数
            for st, ed in _gen_frame_indices( # 每次训练时每个样本最多T=2000的序列，所以这里把全部样本的2000序列都记录一下
                                              # 对应的起始和结束位置，这里为什么要*subsampling呢，因为取样本是基于e.g. Fbanks特征上取，
                                              # 2000是指下采样后结果，而st和ed是采样后的下标，所以要先还原回之前的下标。
                    data_len, chunk_size, chunk_size, use_last_samples,
                    label_delay=self.label_delay,
                    subsampling=self.subsampling):
                self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling))
        print(len(self.chunk_indices), " chunks")

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, i):
        rec, st, ed = self.chunk_indices[i]
        Y, T = feature.get_labeledSTFT(
            self.data,
            rec,
            st,
            ed,
            self.frame_size,
            self.frame_shift,
            self.n_speakers)
        # Y: (frame, num_ceps)
        Y = feature.transform(Y, self.input_transform, sr=self.sr)
        # Y_spliced: (frame, num_ceps * (context_size * 2 + 1))
        Y_spliced = feature.splice(Y, self.context_size)
        # Y_ss: (frame / subsampling, num_ceps * (context_size * 2 + 1))
        Y_ss, T_ss = feature.subsample(Y_spliced, T, self.subsampling)

        Y_ss = torch.from_numpy(Y_ss).float()
        T_ss = torch.from_numpy(T_ss).float()
        return Y_ss, T_ss # 相当于有了每一帧的数据及其对应的speaker标签，每一帧之间相隔（10倍下采样加上80的帧移）100ms
