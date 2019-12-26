#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-12-18 17:05
# @Author  : 冯佳欣
# @File    : data.py.py
# @Desc    : 数据生成器

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO,format = '%(message)s')


class Data_info:
    '''
    该类用于获取保存数据中的一些信息
    '''
    def load_entity(self):
        self.entity_id_map = dict()
        with open(os.path.join(self.data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                vec = line.strip().split('\t')
                self.entity_id_map[vec[0]] = int(vec[1])
        self.id_entity_map = {v: k for k, v in self.entity_id_map.items()}
        self.entity_num = len(self.entity_id_map)
        logging.info('entity_num: %d' % self.entity_num)

    def load_relation(self):
        self.rel_id_map = dict()
        with open(os.path.join(self.data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                vec = line.strip().split('\t')
                self.rel_id_map[vec[0]] = int(vec[1])
        self.id_rel_map = {v: k for k, v in self.rel_id_map.items()}
        self.rel_num = len(self.rel_id_map)
        logging.info('relation_num: %d' % self.rel_num)

    @staticmethod
    def gen_hr_t(triple_data):
        hr_t = dict()
        for h, t, r in triple_data:
            if h not in hr_t:
                hr_t[h] = dict()
            if r not in hr_t[h]:
                hr_t[h][r] = set()
            hr_t[h][r].add(t)
        return hr_t

    @staticmethod
    def gen_tr_h(triple_data):
        tr_h = dict()
        for h,r,t in triple_data:
            if t not in tr_h:
                tr_h[t] = dict()
            if r not in tr_h[t]:
                tr_h[t][r] = set()
            tr_h[t][r].add(h)
        return tr_h

    def load_triple(self,file_path):
        with open(file_path, 'r', encoding='utf-8') as f_triple:
            return np.asarray([[self.entity_id_map[x.strip().split('\t')[0]],
                                self.entity_id_map[x.strip().split('\t')[1]],
                                self.rel_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
                              dtype=np.int32)



    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.load_entity()
        self.load_relation()

        # load train valid test
        self.train_triple = self.load_triple(os.path.join(self.data_dir,'train.txt'))
        logging.info('train triple size : %d'%len(self.train_triple))
        self.valid_triple = self.load_triple(os.path.join(self.data_dir,'valid.txt'))
        logging.info('valid triple size : %d' % len(self.valid_triple))
        self.test_triple = self.load_triple(os.path.join(self.data_dir,'test.txt'))
        logging.info('test triple size : %d' % len(self.test_triple))

        self.train_hr_t = Data_info.gen_hr_t(self.train_triple)
        #self.train_tr_h = Data_info.gen_tr_h(self.train_triple)

        self.hr_t = Data_info.gen_hr_t(np.concatenate([self.train_triple,self.valid_triple,self.test_triple],axis=0))
        #self.tr_h = Data_info.gen_tr_h(np.concatenate([self.train_triple,self.valid_triple,self.test_triple],axis=0))






class ProjE_Dataset(Dataset):
    def get_weight(self, candiate_set):
        return [1. if x in candiate_set else y for
                x, y in
                enumerate(np.random.choice([0., -1.], size=self.entity_num, p=[1 - self.neg_weight, self.neg_weight]))]

    def __init__(self,htr,hr_t,entity_num,neg_weight=0.25):
        self.htr = htr
        self.hr_t = hr_t
        self.entity_num = entity_num
        self.neg_weight=neg_weight

        for idx in range(htr.shape[0]):
            head_id = htr[idx,0]
            tail_id = htr[idx,1]
            rel_id = htr[idx,2]


            # tail rel -> head
            # tr_h_candset = tr_h[tail_id][rel_id]
            # tr_hweight.append(
            #     self.get_weight(tr_h_candset)
            # )
            # tr_hlist.append([tail_id,rel_id])

            # head rel -> tail
            hr_t_candset = hr_t[head_id][rel_id]
            hr_tweight.append(
                self.get_weight(hr_t_candset)
            )
            hr_tlist.append([head_id,rel_id])

        self.hr_t_array = torch.from_numpy(np.asarray(hr_tlist,dtype=np.int32)).long()
        self.hr_tweight_array = torch.from_numpy(np.asarray(hr_tweight,dtype=np.float32))
        # self.tr_h_array = np.asarray(tr_hlist,dtype=np.int32)
        # self.tr_hweight_array = np.asarray(tr_hweight,dtype=np.float32)




    def __len__(self):
        return len(self.htr)


    def __getitem__(self, index):
        head_id = self.htr[index,0]
        tail_id = self.htr[index,1]
        rel_id = self.htr[index,2]
        hr_t_candset = self.hr_t[head_id][rel_id]
        hr_tweight = self.get_weight(hr_t_candset)
        hr_tlist_tensor = torch.Tensor([head_id,rel_id]).long()
        hr_tweight_tensor = torch.Tensor(hr_tweight,dtype=np.float32)
        return hr_tlist_tensor,hr_tweight_tensor


def get_data_loader(hp):
    '''
    :param hp:
        data_dir
        batch_size
        eval_batch

    :return:
    '''
    data_Info = Data_info(hp.data_dir)

    train_data = ProjE_Dataset(data_Info.train_triple,data_Info.train_hr_t,data_Info.entity_num)

    valid_data = ProjE_Dataset(data_Info.valid_triple,data_Info.hr_t,data_Info.entity_num)

    test_data = ProjE_Dataset(data_Info.test_triple, data_Info.hr_t, data_Info.entity_num)

    train_loader = DataLoader(train_data,batch_size = hp.batch_size,shuffle=True,num_workers=2)

    valid_loader = DataLoader(valid_data,batch_size=hp.eval_batch,shuffle=False,num_workers=2)

    test_loader = DataLoader(test_data,batch_size=hp.eval_batch,shuffle=False,num_workers=2)

    return train_loader,valid_loader,test_loader,data_Info.hr_t,data_Info.entity_num,data_Info.rel_num












