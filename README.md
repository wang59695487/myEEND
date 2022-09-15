# How To Use
## Infer a single wav
```
python eend/bin/infer.py -c conf/large/infer.yaml sample/final_4.wav pretrained_models/large/model_callhome.th exp --single --gpu 1
```
> sample/final_4.wav是要预估的音频文件，按需要改成自己的即可。

## Infer wavs in a dir
创建一个测试文件夹，比如test_data/，文件夹下需要有一个wav.scp文件，内容是
```
recid xx/yy/zz.wav
... 
```
则预估：
```
python eend/bin/infer.py -c conf/large/infer.yaml test_data/ pretrained_models/large/model_callhome.th exp --gpu 1
```

## make rttm
推理后处理，并产生每个speaker的音频时间片段文件rttm，其中还会做中值滤滑平滑操作。
```
python eend/bin/make_rttm.py --media=11 --threshold=0.5 --frame_shift=80 --subsampling=10 --sampling_rate=8000 exp/file_list exp/final_4.rttm
```

## Train the model & Prepare data
包括前面的内容，参考run.sh脚本即可。这里主要说明一下，训练数据长什么样子。

准备好训练数据，如下：
```
train_data/
   |
   | ---- segments
   | ---- wav.scp
   | ---- utt2spk
   | ---- rec2dur
   | ---- spk2utt
```
关于每个文件，解释如下：
### wav.scp
每个对话音频的表单文件，每行表示对话id 对应wav文件地址, e.g.
```
0001 xx/yy/zz.wav
0002 xx/yy/kk.wav
...
```

### rec2dur
每个对话音频对应的时长表单
```
0001 200.1
0002 123.2
...
```

### segments
这个是记录每个对话音频中，每句话的起始和结束时间, 每行表示句子编号，以应对话编号，起、终时间，e.g.
```
0001-1 0001 0.000 3.110
0001-2 0001 3.332 6.211
...
```

### utt2spk
这个记录了每个句子是哪个speaker说的，每行表示句子编号和对应的speakerID，e.g.
```
0001-1 3
0001-2 2
...
```

### spk2utt
（可无）这个表单记录的是每个speaker对应的句子编号列表，e.g.
```
spkID utt-id1 utt-id2 utt-id3 ...
...
```

> 其中的句子编号，对话编号，speakerID反正是自己定义的独特的标识即可。

# EEND_PyTorch
A PyTorch implementation of [End-to-End Neural Diarization](https://ieeexplore.ieee.org/document/9003959).

This repo is largely based on the original chainer implementation [EEND](https://github.com/hitachi-speech/EEND) by [Hitachi Ltd.](https://github.com/hitachi-speech), who holds the copyright.

This repo only includes the training/inferring part. If you are looking for data preparation, please refer to the [original authors' repo](https://github.com/hitachi-speech/EEND/blob/master/egs/callhome/v1/run_prepare_shared.sh).

## Note
Only Transformer model with PIT loss is implemented here. And I can only assure the main pipeline is correct. Some side stuffs (such as save_attn_weight, BLSTM model, deep clustering loss, etc.) are either not implemented correctly or not implemented.

Actually the orignal chainer code reserves the pytorch interface, I may consider make a merge request after the code is well-polished.

## Run
1. Prepare your kaldi-style data and modify `run.sh` according to your own directories.
2. Check configuration file. The default `conf/large/train.yaml` configuration uses a 4 layer Transformer with 100k warmsteps, which is different from their paper in ASRU2019. This configuration comes from [their paper submitted to TASLP](https://arxiv.org/abs/2003.02966). As larger model yeilds better performance.
3. `./run.sh`

## Pretrained Models
Pretrained models are offerred here.

`model_simu.th` is trained on simulation data (beta=2), and `model_callhome.th` is adapted on callhome data. They are all 4-layer Transformer models trained with `conf/large/train.yaml`.

## Results
We miss the SwitchBoard Phase 1 for training data, so the results can be a little worse.
| Type | Transformer Layer | Noam Warmup Steps | DER on simu | DER on callhome |
|:-:|:-:|:-:|:-:|:-:|
| [Chainer (ASRU2019)](https://ieeexplore.ieee.org/document/9003959) | 2 | 25k | 7.36 | 12.50 |
| [Chainer (TASLP)](https://arxiv.org/pdf/2003.02966.pdf) | 4 | 100k | 4.56 | 9.54 |
| Chainer (run on our data) | 2 | 25k | 9.78 | 14.85 |
| PyTorch (epoch 50 on simu) | 2 | 25k | 10.14 | 15.72 |
| PyTorch | 4 | 100k | 6.76 | 11.21 |
| PyTorch\* | 4 | 100k | - | 9.35 |

(\* run on full training data, credit to my great colleague!)

## Citation
Cite their great papers!
```
@inproceedings={fujita2019endtoend2,
    title={End-to-End Neural Speaker Diarization with Permutation-Free Objectives},
    author={Fujita, Yusuke and Kanda, Naoyuki and Horiguchi, Shota and Nagamatsu, Kenji and Watanabe, Shinji},
    booktitle={INTERSPEECH},
    year={2019},
    pages={4300--4304},
}
```
```
@inproceedings={fujita2019endtoend,
    title={End-to-End Neural Speaker Diarization with Self-Attention},
    author={Fujita, Yusuke and Kanda, Naoyuki and Horiguchi, Shota and Xue, Yawen and Nagamatsu, Kenji and Watanabe, Shinji},
    booktitle={IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
    pages={296--303},
    year={2019},
}
```
```
@article={fujita2020endtoend,
    title={End-to-End Neural Diarization: Reformulating Speaker Diarization as Simple Multi-label Classification},
    author={Fujita, Yusuke and Watanabe, Shinji and Horiguchi, Shota and Xue, Yawen and Nagamatsu, Kenji},
    journal={arXiv:2003.02966},
    year={2020},
}
```
