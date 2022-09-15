# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
import os
import numpy as np
from tqdm import tqdm
import logging

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from eend.pytorch_backend.models import TransformerModel, NoamScheduler
from eend.pytorch_backend.diarization_dataset import KaldiDiarizationDataset, my_collate
from eend.pytorch_backend.loss import batch_pit_loss, report_diarization_error


def test(args):
    """ Training model with pytorch backend.
    This function is called from eend/bin/train.py with
    parsed command-line arguments.
    """
    # Logger settings====================================================
    formatter = logging.Formatter("[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("Pytorch")
    fh = logging.FileHandler(args.model_save_dir + "/test.log", mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # ===================================================================
    logger.info(str(args))

    np.random.seed(args.seed)
    os.environ['PYTORCH_SEED'] = str(args.seed)

    dev_set = KaldiDiarizationDataset(
        data_dir=args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        )

    # Prepare model
    Y, T = next(iter(dev_set))
    
    if args.model_type == 'Transformer':
        model = TransformerModel(
                n_speakers=args.num_speakers,
                in_size=Y.shape[1],
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                has_pos=False,
                residual=args.rx
                )
    else:
        raise ValueError('Possible model_type is "Transformer"')
    
    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu > 0) else "cpu")
    if device.type == "cuda":
        print('using cuda for testing ....')
        model = nn.DataParallel(model, list(range(args.gpu)))
    model = model.to(device)
    logger.info('Prepared model')
    logger.info(model)

    # Init/Resume
    assert args.initmodel, 'initmodel must be provided during test phase ....'
    if args.initmodel:
        logger.info(f"Load model from {args.initmodel}")
        model.load_state_dict(torch.load(args.initmodel))

    dev_iter = DataLoader(
            dev_set,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=8,
            collate_fn=my_collate
            )

    model.eval()
    with torch.no_grad():
        stats_avg = {}
        cnt = 0
        for y, t in dev_iter:
            y = [yi.to(device) for yi in y]
            t = [ti.to(device) for ti in t]
        
            ilens = [x.shape[0] for x in y]
            ilens_tensor = torch.Tensor(ilens).to(device)
            y = nn.utils.rnn.pad_sequence(y, padding_value=-1, batch_first=True)

            output = model(y, ilens_tensor)
            output = [out[:ilen] for out, ilen in zip(output, ilens)]

            _, label = batch_pit_loss(output, t)
            stats = report_diarization_error(output, label) # stats记录的是平均每个数据的结果
            for k, v in stats.items():
                stats_avg[k] = stats_avg.get(k, 0) + v
            cnt += 1
        stats_avg = {k:v/cnt for k,v in stats_avg.items()}
        # （漏帧+多召帧+错帧) / 有人说话的总帧数（包括重叠部分），---> 这样看DER是有可能大于1的。
        stats_avg['DER'] = stats_avg['diarization_error'] / stats_avg['speaker_scored'] * 100
        for k in stats_avg.keys():
            stats_avg[k] = round(stats_avg[k], 2)

        logger.info(f"Dev Stats: {stats_avg}")

    logger.info('Finished!')
