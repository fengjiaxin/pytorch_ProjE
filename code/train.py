#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-12-25 19:41
# @Author  : 冯佳欣
# @File    : train.py
# @Desc    : 训练函数

import torch
from ProjE_sigmoid import ProjE
from data import get_data_loader
from hparams import Hprams
import time
import numpy as np
import math
import os
import logging
logging.basicConfig(level=logging.INFO,format = '%(message)s')

use_gpu = torch.cuda.is_available()

# 评估数据rank
def evaluation_helper(test_htr,tail_pred,hr_t):
    '''
    :param test_htr: [batch,3] 
    :param tail_pred: [batch,entity_num]
    :param hr_t: 
    :return: 
    '''
    assert len(test_htr) == len(tail_pred)
    entity_num = tail_pred.shape[1]
    _,tail_ids = tail_pred.topk(k=entity_num)
    #mean_rank_h = list()
    mean_rank_t = list()
    #filtered_mean_rank_h = list()
    filtered_mean_rank_t = list()


    for i in range(test_htr.shape[0]):
        h = test_htr[i,0]
        t = test_htr[i,1]
        r = test_htr[i,2]

        # mean rank
        # mr = 0
        # for val in head_pred[i]:
        #     if val == h:
        #         mean_rank_h.append(mr)
        #         break
        #     mr += 1
        mr = 0
        for val in tail_ids[i]:
            if int(val) == t:
                mean_rank_t.append(mr)
                break
            mr += 1

        # filtered mean rank
        # fmr = 0
        # for val in head_pred[i]:
        #     if val == h:
        #         filtered_mean_rank_h.append(fmr)
        #         break
        #     if t in tr_h and r in tr_h[t] and val in tr_h[t][r]:
        #         continue
        #     else:
        #         fmr += 1

        fmr = 0
        for val in tail_ids[i]:
            if int(val) == t:
                filtered_mean_rank_t.append(fmr)
                break
            if h in hr_t and r in hr_t[h] and val in hr_t[h][r]:
                continue
            else:
                fmr += 1

    return (mean_rank_t,filtered_mean_rank_t)



def val(model,data_loader,hp,hr_t):
    '''
    评估模型在
    :param model:
    :param data_loader:
        hr_tlist
        hr_tweight
    :param hp:
    :return:
        tail mean rank
        tail filtered mean rank
        mean rank hit@10
        filtered mean rank hit@10
    '''
    # 将模型设置为val模式
    model.eval()

    accu_mean_rank_t_list = list()
    accu_filtered_mean_rank_t_list = list()

    for (batch_htr,batch_hr_tlist,_) in data_loader:
        if hp.cuda:
            batch_htr = batch_htr.cuda()
            batch_hr_tlist = batch_hr_tlist.cuda()
        pred_tail = model(batch_hr_tlist)
        mrt,fmrt = evaluation_helper(batch_htr,pred_tail,hr_t)

        accu_mean_rank_t_list += mrt
        accu_filtered_mean_rank_t_list += fmrt

    return np.mean(accu_mean_rank_t_list),np.mean(accu_filtered_mean_rank_t_list),\
           np.mean(np.asarray(accu_mean_rank_t_list,dtype=np.int32) < 10),\
           np.mean(np.asarray(accu_filtered_mean_rank_t_list,dtype=np.int32) < 10)

def batch_train(model,b_hr_tlist,b_hr_tweight,optimizer,clip):
    model.train()

    optimizer.zero_grad()
    hrt_res = model(b_hr_tlist)
    loss = model.neg_loss(hrt_res,b_hr_tweight)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
    optimizer.step()
    #logging.info('loss:%s'%str(loss.item()))
    return loss.item()

# 显示时间
def timeSince(since, percent):
    '''
    :param since: 开始记录的time时刻
    :param percent: 已完成的百分比
    :return:
    '''
    now = time.time()
    pass_time = now - since
    all_time = pass_time / percent
    remain_time = all_time - pass_time
    return '%s (- %s)' % (asMinutes(pass_time), asMinutes(remain_time))

def asMinutes(s):
    '''
    将时间s转换成minute 和 second的组合
    :param s:
    :return:
    '''
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def main():
    hparams = Hprams()
    parser = hparams.parser
    hp = parser.parse_args()

    # 是否可以使用GPU
    if not use_gpu:
        setattr(hp, "cuda", False)
        setattr(hp, "device", -1)
        torch.manual_seed(hp.seed)
    train_iter,valid_iter,test_iter,hr_t,entity_num,rel_num = get_data_loader(hp)
    # 更新hp
    setattr(hp, "entity_num", entity_num)
    setattr(hp, "relation_num", rel_num)

    # 初始化模型
    model = ProjE(hp)
    optimizer = torch.optim.Adam(model.parameters(),lr=hp.lr,weight_decay=hp.weight_decay)

    # 模型保存目录
    model_dir = '../save_models/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if hp.cuda:
        torch.cuda.manual_seed(hp.seed)
        model.cuda()

    # 总batch 批数
    n_iters = hp.max_epoches * len(train_iter)
    curr_iter = 0

    # 随机初始化 valid的rank结果
    init_valid_mean_rank,init_valid_filtered_mean_rank,init_valid_hit,init_valid_filtered_hit = val(model,valid_iter,hp,hr_t)
    init_valid_str = " valid init [tail pred] mean rank: %.1f filtered_mean_rank: %.1f hit@10: %.3f filtered hit@10: %.3f"\
                     %(init_valid_mean_rank,init_valid_filtered_mean_rank,init_valid_hit,init_valid_filtered_hit)
    logging.info(init_valid_str)

    # 随机初始化 test的rank结果
    init_test_mean_rank,init_test_filtered_mean_rank,init_test_hit,init_test_filtered_hit = val(model,test_iter,hp,hr_t)
    init_test_str = " test  init [tail pred] mean rank: %.1f filtered_mean_rank: %.1f hit@10: %.3f filtered hit@10: %.3f"\
                     %(init_test_mean_rank,init_test_filtered_mean_rank,init_test_hit,init_test_filtered_hit)
    logging.info(init_test_str)

    start_time = time.time()
    for epoch in range(hp.max_epoches):
        # 这个是一个轮次的数据统计
        total_loss = 0.0
        total_num = len(train_iter)

        # 这个是打印时候的数据统计
        print_total_loss = 0.
        print_total_num = 0

        for idx,(_,b_hr_tlist,b_hr_tweight) in enumerate(train_iter):
            batch_len = b_hr_tlist.shape[1]
            batch_loss = batch_train(model,b_hr_tlist,b_hr_tweight,optimizer,hp.clamp)

            print_total_loss += batch_loss
            print_total_num += batch_len
            total_loss += batch_loss
            curr_iter += 1

            if idx+1 % hp.print_every == 0:
                #logging.info('print total loss:%s'%str(print_total_loss))
                #logging.info('print total num:%s' % str(print_total_num))
                logging.info("epoch:%d,batch_id:%d,time:%s (%d %d%%) loss:%.4f"%
                             (epoch + 1,idx + 1,timeSince(start_time,(idx+1)/n_iters),curr_iter,curr_iter/n_iters * 100,print_total_loss/print_total_num))
                print_total_loss = 0.
                print_total_num = 0

        # 评估epoch的损失
        logging.info('[train epoch: %d] avg_loss:%.4f'%(epoch + 1,total_loss/total_num))

        # 迭代一个轮次，进行valid和test 的 rank 结果
        #===========valid================
        valid_mean_rank, valid_filtered_mean_rank, valid_hit, valid_filtered_hit = val(model,valid_iter,hp, hr_t)
        valid_str = "train epoch:%d valid [tail pred] mean rank: %.1f filtered_mean_rank: %.1f hit@10: %.3f filtered hit@10: %.3f" \
                         % (epoch + 1,valid_mean_rank, valid_filtered_mean_rank, valid_hit,valid_filtered_hit)
        logging.info(valid_str)
        # ==========test=================
        test_mean_rank, test_filtered_mean_rank, test_hit, test_filtered_hit = val(model,test_iter,hp, hr_t)
        test_str = "train epoch:%d test  [tail pred] mean rank: %.1f filtered_mean_rank: %.1f hit@10: %.3f filtered hit@10: %.3f" \
                         % (epoch + 1,test_mean_rank, test_filtered_mean_rank, test_hit,test_filtered_hit)
        logging.info(test_str)

    #     # 模型保存文件
    save_path = os.path.join(model_dir,'{}.pth'.format(hp.model_id))
    final_model = {
        'state_dict':model.state_dict(),
        'config':vars(hp)
    }
    torch.save(final_model,save_path)
    logging.info('final model saved in {}'.format(save_path))

if __name__ == '__main__':
    main()




















