# LearnToPayAttention  

[![AUR](https://img.shields.io/aur/license/yaourt.svg?style=plastic)](LICENSE)   

PyTorch implementation of ICLR 2018 paper [Learn To Pay Attention](http://www.robots.ox.ac.uk/~tvg/publications/2018/LearnToPayAttention_v5.pdf)  

![](https://github.com/SaoYan/LearnToPayAttention/blob/master/fig/learn_to_pay_attn.png)

My implementation is based on "(VGG-att3)-concat-pc" in the paper, and I trained the model on CIFAR-100 DATASET.  
I implemented two version of the model, the only difference is whether to insert the attention module before or after the corresponding max-pooling layer.

## Training  
- python3 ./train.py --cuda True
- python3 ./train.py --cuda True --attn_before False

## Results  
- checkpoints/results can see them
