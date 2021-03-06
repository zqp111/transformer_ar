# -*- encoding: utf-8 -*-
'''
@File    :   IO.py
@Time    :   2020/09/11 15:28:55
@Author  :   zqp 
@Version :   1.0
@Contact :   zhangqipeng@buaa.edu.cn
'''

import argparse
import sys
import unittest
from processor.base_method import import_class, str2bool
import yaml
import pickle
import torch
from collections import OrderedDict
import os
import torch.nn as nn
from processor.mylog import MyLog
import time


class IO():
    """
    Implementation of
    """
    def __init__(self, args=None):
        self.load_args(args)
        self.init_log()
        self.init_env()
        self.load_model()
        # self.load_weights()
        


    def load_args(self, args=None):
        parser = self.get_parser()
        p = parser.parse_args(args)
        if p.config is not None:
            with open(p.config, 'r') as f:
                default_args = yaml.load(f)
        keys = vars(p).keys()
        for k in default_args.keys():
            assert k in keys, "Unknown configuration {}".format(k)
        parser.set_defaults(**default_args)

        self.args = parser.parse_args(args)

    def init_env(self):
        # 设置GPU卡数
        gpus = [self.args.device] if type(self.args.device) is int else list(self.args.device)
        # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(list(map(str, gpus)))
        if type(self.args.device) is int:
            self.output_device =self.args.device
        else:
            self.output_device =self.args.device[0]
        self.logger.log("The GPU devices:{} are used.".format(self.args.device))
        self.logger.log("The output devices is GPU:{} ".format(self.output_device))

    def load_model(self):
        # 加载模型
        Model = import_class(self.args.model)
        self.model = Model(**self.args.model_args).cuda(self.output_device)
        self.load_weights()

        if type(self.args.device) is list:
            if len(self.args.device) > 1:
                self.model = nn.DataParallel(self.model, 
                                             device_ids=self.args.device, 
                                             output_device=self.output_device)
        self.logger.log("The Model is {}".format(self.model))

    def load_weights(self):
        if self.args.weights == None:
            print("No weights file need to load")  
            self.logger.log("Dont load weights file")
        else:
            if ".pkl" in self.args.weights:     
                with open(self.args.weights) as f:
                    weights = pickle.load(f)
            elif ".pt" in self.args.weights:
                weights = torch.load(self.args.weights)
            else:
                raise Exception("The weights file cant be load")
            
            weights = OrderedDict([(k.split("model.")[-1], v.cuda(self.output_device)) for k, v in weights.items()])
            for w in self.args.ignore_weights:
                if weights.pop(w, None) is not None:
                    print("success to remove weights:{}".format(w))
                    self.logger.log("success to remove weights:{}".format(w))
                else:
                    raise Exception("fail to remove weights:{}".format(w))
            
            try:
                self.model.load_state_dict(weights)
                self.logger.log("success load weights from {}".format(self.args.weights))
            except Exception:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print("cant find these weights:", diff)

    def init_log(self):
        dir_time = time.strftime('%Y-%m-%d', time.localtime())
        log_path = os.path.join(self.args.work_dir,self.args.Experiment_name+dir_time)
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.logger = MyLog(log_path)
        self.logger.log("success to init logger")

    def save_model(self, epoch):
        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
        weights_save_path = os.path.join(self.log_path, "save_models")
        if not os.path.exists(weights_save_path):
            os.mkdir(weights_save_path)
        torch.save(weights, weights_save_path + "/" + self.args.Experiment_name + '-' + str(epoch)+ '.pt')


    @staticmethod
    def get_parser(add_help=False):
        parser = argparse.ArgumentParser(add_help=add_help, description="IO processor")

        # model
        parser.add_argument("--Experiment_name",default="GCN",help="The trained network name")
        parser.add_argument("--model", default="model.model.Model", help="The model name")
        parser.add_argument("--model_args", default=dict(), help="The arguments of model")
        parser.add_argument('--work_dir', default="./work_dir/", help="The dir of the result and arg from trained model")
        parser.add_argument('--config', default="./test.yaml", help="The config of model")
        parser.add_argument('--ignore_args', default=dict(), help="The arguments to ignore when load weights")
        parser.add_argument('--weights', default=None, help="the saved weights")
        parser.add_argument('--ignore_weights', default=dict(), help="the arguments to ignore when load weights")
        # feeder
        parser.add_argument('--feeder', default="feeders.feeders.Feeder", help="The data feeder name")
        parser.add_argument('--train_feeder_args', default=dict(), help="The args of the train feeder")
        parser.add_argument('--test_feeder_args', default=dict(), help="The args of the test feeder")
        parser.add_argument("--num_workers", type=int, default=2, help="The number of thread")

        # debug
        parser.add_argument("--print_log", type=str2bool, default=True, help="whether print log information")
        parser.add_argument("--debug", type=str2bool, default=False, help="whether to debug")

        # train parameter
        parser.add_argument("--warm_up_epoch", type=int, default=0, help="the epoch number to use base learning rate")
        parser.add_argument("--epochs", type=int, default=100, help="The num of epochs to train")
        parser.add_argument("--only_train_epoch", type=int, default=0, help="Some parameters are only train after this epoch")
        parser.add_argument("--step", type=int, default=[20, 50, 80], help="The number epoch to decrease the learning rate")
        parser.add_argument("--optim", type=str, default="Adam", help="The optimizer to use")
        parser.add_argument("--device", type=int, default=[0, 1, 2, 3], help="The CUDA device to use")
        parser.add_argument("--lr", type=float, default=0.001, help="The number of learning rate")
        parser.add_argument("--weight_decay", type=float, default=0.0001, help="The weight decay")
        parser.add_argument("--batch_size", type=int, default=32, help="The batch size")
        parser.add_argument("--phase", default="train", help="whether to train")
        parser.add_argument("--nesterov", type=str2bool, default=True, help="When use SGD optim is required")
        parser.add_argument("--log_interval", type=int, default=50, help='')
        parser.add_argument("--save_interval", type=int, default=1, help="")
        parser.add_argument("--save_results", type=str2bool, default=True, help="wether to save results")

        return parser


if __name__ == "__main__":
    a = IO()