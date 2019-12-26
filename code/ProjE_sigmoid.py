#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-12-25 11:49
# @Author  : 冯佳欣
# @File    : ProjE_sigmoid.py
# @Desc    : pointwise 预测  为了方便程序编码 ，这里只采用 head rel -> tail

import torch
import numpy as np
from torch import nn
import math
import logging
logging.basicConfig(level=logging.INFO,format = '%(message)s')

class ProjE(nn.Module):
    def __init__(self,hp):
        '''
        初始化
        :param hp:
            需要的参数如下
            embed_dim
            dropout_rate
            entity_num
            relation_num
            reglarizer_weight
        '''
        super(ProjE,self).__init__()
        self.embed_dim = hp.embed_dim
        self.dropout_rate = hp.dropout_rate
        self.entity_num = hp.entity_num
        self.relation_num = hp.relation_num
        # 正则化权重
        self.regularizer_weight = hp.regularizer_weight

        # 权重初始化
        bound = 6 / math.sqrt(self.embed_dim)

        # 实体矩阵初始化
        self.entity_embedding = nn.Embedding(self.entity_num,self.embed_dim)
        nn.init.uniform_(self.entity_embedding.weight, a=-bound, b=bound)

        # 关系矩阵初始化
        self.rel_embedding = nn.Embedding(self.relation_num,self.embed_dim)
        nn.init.uniform_(self.rel_embedding.weight, a=-bound, b=bound)

        # global vector
        self.hr_weighted_vector = nn.Parameter(torch.FloatTensor(self.embed_dim * 2))
        #self.tr_weighted_vector = nn.Parameter(torch.FloatTensor(self.embed_dim * 2))
        nn.init.uniform_(self.hr_weighted_vector, a=-bound, b=bound)
        #nn.init.uniform_(self.tr_weighted_vector, a=-bound, b=bound)

        # global bias
        self.hr_combination_bias = nn.Parameter(torch.FloatTensor(self.embed_dim))
        #self.tr_combination_bias = nn.Parameter(torch.FloatTensor(self.embed_dim))
        nn.init.uniform_(self.hr_combination_bias, a=-bound, b=bound)
        #nn.init.uniform_(self.tr_combination_bias, a=-bound, b=bound)

        self.hrt_dropout = nn.Dropout(self.dropout_rate)
        #self.trh_dropout = nn.Dropout(self.dropout_rate)



    def forward(self,hr_tlist):
        '''
        给定hr 和 weight ,预测接下来每个实体的分数
        :param inputs:
        # hr_tlist : [batch,2],
        :return:
        hrt_res [batch,entity_num]
        '''

        # [batch,dim]
        hr_tlist_h = self.entity_embedding(hr_tlist[:,0])
        hr_tlist_r = self.rel_embedding(hr_tlist[:,1])

        #tr_hlist_t = self.entity_embedding(tr_hlist[:,0])
        #tr_hlist_r = self.rel_embedding(tr_hlist[:,1])

        # predict tail
        # 点乘 [batch,dim]
        hr_tlist_hr = hr_tlist_h * self.hr_weighted_vector[:self.embed_dim] + \
                      hr_tlist_r * self.hr_weighted_vector[self.embed_dim:]
        hrt_res = torch.matmul(self.hrt_dropout(torch.tanh(hr_tlist_hr + self.hr_combination_bias)),self.entity_embedding.weight.t())

        # predict head
        #tr_hlist_hr = tr_hlist_t * self.tr_weighted_vector[:self.embed_dim] + \
        #    tr_hlist_r * self.tr_weighted_vector[self.embed_dim:]
        #trh_res = torch.matmul(self.trh_dropout(torch.tanh(tr_hlist_hr + self.tr_combination_bias)),self.entity_embedding.t())

        return hrt_res

    def neg_loss(self,hrt_res,hr_tlist_weight):
        '''
        返回损失函数
        weight列表中 1代表正例 -1代表负例 0代表未被采样
        # hrt_res : [batch,entity_num]
        # hr_tlist_weight : [batch,entity_num]
        :return:
        loss
        '''

        # 正则化损失
        regularizer_loss = torch.sum(torch.abs(self.hr_weighted_vector)) + \
            torch.sum(torch.abs(self.entity_embedding.weight)) + \
            torch.sum(torch.abs(self.rel_embedding.weight))

        # hrt
        hrt_res_sigmoid = torch.sigmoid(hrt_res)

        # log(h(e,r)i)
        x_ = torch.log(torch.clamp(hrt_res_sigmoid, 1e-10, 1.0)) * torch.clamp(hr_tlist_weight.float(),min=0.0)
        x_sum = torch.sum(x_)
        #logging.info('x_sum:%s' % str(x_sum))

        # log(1-h(e,r)i)
        y_ = torch.log(torch.clamp(1. - hrt_res_sigmoid , 1e-10, 1.0)) * torch.clamp(-hr_tlist_weight.float(),min=0.0)
        y_sum = torch.sum(y_)
        #logging.info('y_sum:%s' % str(y_sum))

        hrt_loss = -x_sum - y_sum
        #logging.info('hrt_loss:%s' % str(hrt_loss))

        #logging.info('regular_loss:%s'%str(regularizer_loss))

        #return hrt_loss + regularizer_loss * self.regularizer_weight

        return hrt_loss











