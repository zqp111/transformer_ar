import torch.nn as nn
import torch
from tqdm import tqdm
from processor.process import Process
from processor.base_method import import_class
import time
import numpy as np

class Recognition(Process):

    def load_model(self):
        # 加载模型
        Model = import_class(self.args.model)
        self.model = Model(**self.args.model_args).cuda(self.output_device)
        self.load_weights() # 在这里加载权重 避免多卡单卡保存的权重名称不一问题

        if type(self.args.device) is list:
            if len(self.args.device) > 1:
                self.model = nn.DataParallel(self.model, 
                                             device_ids=self.args.device, 
                                             output_device=self.output_device)
        self.logger.log("The Model is {}".format(self.args.model))

        # 加载Loss函数
        self.loss = nn.CrossEntropyLoss()

    def get_topk(self, k, result, label):
        rank = result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        return accuracy

    def train(self):
        self.model.train()
        epoch = self.meta_info['epoch']

        self.adjust_learning_rate(epoch)
        self.only_train_part("DecoupleA", epoch)  # Some parameters dont requid grad

        loader = self.data_loader['train']
        loss_value = []
        result_frag = []
        label_frag = []
        self.record_time()  # 记录时间
        process = tqdm(loader, ncols=90)

        for batch_idx, (data, label, index) in enumerate(process):
            # load data
            data = data.float().cuda(self.output_device)
            label = label.long().cuda(self.output_device)
            # forward
            start = time.time()
            output = self.model(data)
            result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())
            forward_time = time.time() - start
            loss = self.loss(output, label)
            # backward
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # record
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = self.optim.param_groups[0]['lr']
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            if batch_idx % self.args.log_interval == 0:
                process.set_description(
                    '\tLoss: {:.4f}  lr:{:.6f}'.format(self.iter_info['loss'], self.iter_info['lr']))


        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        # log
        result_ = np.concatenate(result_frag)
        label_ = np.concatenate(label_frag)
        acc_top1 = self.get_topk(1, result_, label_)
        acc_top5 = self.get_topk(5, result_, label_)
        epoch_time = self.split_time()
        self.logger.log("The train epoch {} time:{}, top1: {}, top5: {}, loss: {}".format(
            epoch, epoch_time, acc_top1, acc_top5, self.epoch_info['mean_loss']))
        print("The train epoch {} time:{}, top1: {}, top5: {}, loss: {}".format(
            epoch, epoch_time, acc_top1, acc_top5, self.epoch_info['mean_loss']))

    def test(self):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        process = tqdm(loader, ncols=90)
        self.record_time()  # 记录时间
        for data, label, _ in process:
            # get data
            data = data.float().to(self.output_device)
            label = label.long().to(self.output_device)
            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())
            # get loss
            loss = self.loss(output, label)
            loss_value.append(loss.item())
            label_frag.append(label.data.cpu().numpy())

        result_ = np.concatenate(result_frag)
        label_ = np.concatenate(label_frag)
        self.epoch_info['mean_loss']= np.mean(loss_value)
        acc_top1 = self.get_topk(1, result_, label_)
        acc_top5 = self.get_topk(5, result_, label_)
        epoch_time = self.split_time()
        self.logger.eval_log("The test epoch {} time:{}, top1: {}, top5: {}, loss: {}".format(
            self.meta_info['epoch'], epoch_time, acc_top1, acc_top5, self.epoch_info['mean_loss']))
        print("The test epoch {} time:{}, top1: {}, top5: {}, loss: {}".format(
            self.meta_info['epoch'], epoch_time, acc_top1, acc_top5, self.epoch_info['mean_loss']))

        return result_, acc_top1
            
        




