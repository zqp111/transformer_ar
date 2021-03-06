from processor.IO import IO
import torch
from processor.base_method import import_class, str2bool
import torch.optim  as optim 
import yaml
import numpy as np
import time
import os
import pickle


class Process(IO):
    def __init__(self, args=None):
        self.load_args(args)
        self.init_log()
        self.init_env()
        self.load_model()
        # self.load_weights()
        self.save_args()
        self.load_data()
        self.load_optim()


    def init_env(self):
        super().init_env()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)   
    
    def save_args(self):
        arg_dict = vars(self.args)
        with open('{}/config.yaml'.format(self.log_path), 'w') as f:
            yaml.dump(arg_dict, f)

    def load_data(self):
        Feeder = import_class(self.args.feeder)
        if "debug" not in self.args.test_feeder_args:
            self.args.test_feeder_args["debug"] = self.args.debug
        self.data_loader = {}
        if self.args.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.args.train_feeder_args),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=True
            )
        self.data_loader["test"] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.args.test_feeder_args),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=True
            )

    def load_optim(self):
        if self.args.optim == "Adam":
            self.optim = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        elif self.args.optim == "SGD":
            self.optim = optim.SGD(self.model.parameters(),
                                   lr=self.args.lr,
                                   momentum=0.9,
                                   weight_decay=self.args.weight_decay,
                                   nesterov=self.args.nesterov)
        else:
            raise ValueError("No match optim to use")

    def adjust_learning_rate(self, epoch):
        if self.args.optim == 'SGD' or self.args.optim == 'Adam':
            if epoch < self.args.warm_up_epoch:
                lr = self.args.lr * (epoch + 1) / self.args.warm_up_epoch
            else:
                lr = self.args.lr * (
                        0.1 ** np.sum(epoch >= np.array(self.args.step)))
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.logger.train_log('\t{}: {}'.format(k, v))

    def show_iter_info(self):
        if self.meta_info['iter'] % self.args.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.logger.train_log(info)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def only_train_part(self, key_words, epoch):
        if epoch >= self.args.only_train_epoch:
            self.logger.log('only train part, require grad')
            for key, value in self.model.named_parameters():
                if key_words in key:
                    value.requires_grad = True
                    # self.logger.log(key + '-require grad')
        else:
            self.logger.log('only train part, do not require grad')
            for key, value in self.model.named_parameters():
                if key_words in key:
                    value.requires_grad = False
                    self.logger.log(key + '-not require grad')

    def save_results(self, results, filename):
        results_path = os.path.join(self.log_path, "results")
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        with open('{}/{}'.format(results_path, filename), 'wb') as f:
            pickle.dump(results, f)


    def train(self):    # The train template
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

        return self.epoch_info['mean loss'] # ???????????????????????????result???acc

    def start(self):  #?????????????????????
        self.logger.log('Parameters:\n{}\n'.format(str(vars(self.args))))
        self.best_score = 0

        # training phase
        if self.args.phase == 'train':
            for epoch in range(self.args.epochs):# TODO: check point
                self.meta_info['epoch'] = epoch
                # training
                self.logger.train_log('Training epoch: {}'.format(epoch))
                self.train()
                self.logger.train_log('Done.\n')
                # save model
                if ((epoch + 1) % self.args.save_interval == 0) or (
                        epoch + 1 == self.args.num_epoch):
                    self.save_model(epoch)
                # evaluation
                self.logger.eval_log('Eval epoch: {}'.format(epoch))
                result, acc = self.test()
                self.logger.eval_log('Done.\n')
                # save the output of model
                result_dict = dict(
                        zip(self.data_loader['test'].dataset.sample_name, result))
                self.save_results(result_dict, 'result_{}_{}.pkl'.format(epoch, acc))
                # save best results
                if acc > self.best_score:
                    self.save_results(result_dict, 'result_best_{}.pkl'.format(acc))



        # test phase
        elif self.args.phase == 'test':
            # the path of weights must be appointed
            if self.args.weights is None:
                raise ValueError('Please appoint --weights.')
            self.logger.log('Model:   {}.'.format(self.args.model))
            self.logger.log('Weights: {}.'.format(self.args.weights))
            # evaluation
            self.logger.eval_log('Evaluation Start:')
            result, acc = self.test()
            self.logger.eval_log('Done.\n')
            # save the output of model
            result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name, result))
            self.save_results(result_dict, 'result_{}_{}.pkl'.format(epoch, acc))
            # save best results
            if acc > self.best_score:
                self.save_results(result_dict, 'result_best_{}.pkl'.format(acc))
        

