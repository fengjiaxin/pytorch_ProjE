#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-12-26 10:29
# @Author  : 冯佳欣
# @File    : hparams.py
# @Desc    : 超参数设置

import argparse

class Hprams:
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../mini/', help='data dir')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--embed_dim', default=100, type=int, help='entity,relation embedding dim')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--eval_batch', default=256, type=int, help='eval batch size')
    parser.add_argument('--print_every', default=1, type=int, help='print every batch')
    parser.add_argument('--model_id', default='sigmoid_0', help='model id')

    parser.add_argument('--entity_num', default=1156, type=int, help='entity num')
    parser.add_argument('--relation_num', default=231, type=int, help='relation num')

    parser.add_argument('--seed',default=77,type=int,help='random seed number')
    parser.add_argument('--device', default=2, type=int, help='gpu device id')
    parser.add_argument('--clamp',default=1.,type=float,help='模型参数更新限制')
    parser.add_argument('--regularizer_weight',default=1.,type=float,help='regularizer weight')
    parser.add_argument('--weight_decay', default=0.9, type=float, help='权重衰减')
    parser.add_argument('--max_epoches',default=10,type=int,help='train epoches')
    parser.add_argument('--cuda',default=True,help='是否可以使用nvida gpu训练,upadte later')