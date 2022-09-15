#!/usr/bin/env python3
#
# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os
import h5py
import numpy as np
from scipy.ndimage import shift
from tqdm import tqdm

import torch
import torch.nn as nn

import soundfile as sf
import librosa

from eend.pytorch_backend.models import TransformerModel
from eend import feature
from eend import kaldi_data


def _gen_chunk_indices(data_len, chunk_size, step=None):
    if step is None:
        step = chunk_size
    start = 0
    while start < data_len:
        end = min(data_len, start + chunk_size)
        yield start, end
        start += step
        # for the last chunksize, no overlap used
        if start + 100 >= data_len:
            start += 100


def infer(args):
    # Prepare model
    # in_size 是用来提前根据参数把网络接收的特征维度先计算出来，作为model的参数传入
    in_size = feature.get_input_dim(
            args.frame_size,
            args.context_size,
            args.input_transform)

    print('init model...')
    if args.model_type == 'Transformer':
        model = TransformerModel(
                n_speakers=args.num_speakers,
                in_size=in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                has_pos=False,
                residual=args.rx
                )
    elif args.model_type == 'TransformerEDA':
         model = TransformerEDADiarization(
            device=args.device,
            in_size=args.feature_dim * (1 + 2 * args.context_size),
            n_units=args.hidden_size,
            e_units=args.encoder_units,
            n_heads=args.transformer_encoder_n_heads,
            n_layers=args.transformer_encoder_n_layers,
            dropout=args.transformer_encoder_dropout,
            attractor_loss_ratio=args.attractor_loss_ratio,
            attractor_encoder_dropout=args.attractor_encoder_dropout,
            attractor_decoder_dropout=args.attractor_decoder_dropout,
            detach_attractor_loss=args.detach_attractor_loss,
         )
    else:
        raise ValueError('Unknown model type.')
    
    
    print('model to dp...')
    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu > 0) else "cpu")
    if device.type == "cuda":
        model = nn.DataParallel(model, list(range(args.gpu)))
    model = model.to(device)

    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    if args.single:
        import time
        st = time.time()
        wav_filepath = args.data_dir
        print(f'processing ...{wav_filepath}')
        data, rate = librosa.load(wav_filepath, sr=args.sampling_rate)
        print(time.time() - st)
        st = time.time()
        print(data.shape, rate)
        #data, rate = sf.read(wav_filepath, start=0, stop=None)
        Y = feature.stft(data, args.frame_size, args.frame_shift) # 对原始数据进行短时傅立叶变换
        Y = feature.transform(Y, transform_type=args.input_transform, sr=args.sampling_rate) # 提取log mel特征，返回的结果是n_frames x n_featdim(nx23)
        Y = feature.splice(Y, context_size=args.context_size) # 这里是对每一帧数据23维扩展成23*15维（是把前后7帧的数据搞到一起了）
        Y = Y[::args.subsampling] # 帧移80个数据，下采样10，即每800个数据采样一次，按8K的采样率，即每100ms采样一个数据，与论文一致； 
        out_chunks = []
        with torch.no_grad():
            hs = None
            for start, end in _gen_chunk_indices(len(Y), args.chunk_size): # 一个chunck有2000个数据，即end-start<=2000，算是T=2000？
                Y_chunked = torch.from_numpy(Y[start:end]) # 2000 x 345
                Y_chunked.to(device)
                ys = model(Y_chunked.unsqueeze(0), activation=torch.sigmoid) # 得到一个chunck也就是2000帧数据的对应结果2000 x S(S是指有S个说话人)
                out_chunks.append(ys[0].cpu().detach().numpy())
                if args.save_attention_weight == 1:
                    raise NotImplementedError()
        outfname = os.path.basename(wav_filepath).split('.')[0] + '.h5'
        outpath = os.path.join(args.out_dir, outfname)
        if args.label_delay != 0:
            outdata = shift(np.vstack(out_chunks), (-args.label_delay, 0))
        else:
            outdata = np.vstack(out_chunks)
        with h5py.File(outpath, 'w') as wf:
            wf.create_dataset('T_hat', data=outdata)
        print(f'Save results in {outpath}')        
        print(time.time() - st)
        return
    
    
    wav_filepaths = os.path.join(args.data_dir)
    wav_files = [] 
    with open(wav_filepaths, 'r') as fr:
        for line in fr.readlines():
            spk, utt = line.strip().split()
            wav_files.append(utt)

    for wav_filepath in tqdm(wav_files):

        data, rate = librosa.load(wav_filepath, sr=args.sampling_rate)
        print(data.shape, rate)
        #data, rate = sf.read(wav_filepath, start=0, stop=None)
        Y = feature.stft(data, args.frame_size, args.frame_shift) # 对原始数据进行短时傅立叶变换
        Y = feature.transform(Y, transform_type=args.input_transform, sr=args.sampling_rate) # 提取log mel特征，返回的结果是n_frames x n_featdim(nx23)
        Y = feature.splice(Y, context_size=args.context_size) # 这里是对每一帧数据23维扩展成23*15维（是把前后7帧的数据搞到一起了）
        Y = Y[::args.subsampling] # 帧移80个数据，下采样10，即每800个数据采样一次，按8K的采样率，即每100ms采样一个数据，与论文一致； 
        out_chunks = []
        with torch.no_grad():
            hs = None
            for start, end in _gen_chunk_indices(len(Y), args.chunk_size, step=args.chunk_size-100): # 一个chunck有2000个数据，即end-start<=2000，算是T=2000？
                Y_chunked = torch.from_numpy(Y[start:end]) # 2000 x 345
                Y_chunked.to(device)
                ys = model(Y_chunked.unsqueeze(0), activation=torch.sigmoid) # 得到一个chunck也就是2000帧数据的对应结果2000 x S(S是指有S个说话人)
                out_chunks.append(ys[0].cpu().detach().numpy())
                if args.save_attention_weight == 1:
                    raise NotImplementedError()
        outfname = os.path.basename(wav_filepath).split('.')[0] + '.h5'
        outpath = os.path.join(args.out_dir, outfname)
        if args.label_delay != 0:
            outdata = shift(np.vstack(out_chunks), (-args.label_delay, 0))
        else:
            outdata = np.vstack(out_chunks)
        with h5py.File(outpath, 'w') as wf:
            wf.create_dataset('T_hat', data=outdata)
        print(f'Save results in {outpath}')        
    