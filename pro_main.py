import torch.nn as nn
import torch
import numpy as np
import argparse
from processor.base_method import import_class
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MuiltKernelGTCN network")

    processors = dict()
    processors['recognition'] = import_class('processor.action_recognition.Recognition')

    subparsers = parser.add_subparsers(dest='processor')  #启动命名空间为processor的添加子命令，dest=‘’将存储子命令名称的属性的名称为processor
    for k, p in processors.items():  #将字典中的每一对变成元组的形式（[name,zhang],[age,20]）
        subparsers.add_parser(k, parents=[p.get_parser()]) #添加子命令K,  这个子命令K继承了p.get_paeser()中定义的所有的命令参数
    # read arguments
    arg = parser.parse_args() #开始读取命令行的数值并保存
    # start
    
    Processor = processors['recognition']  #读取arg的processor属性,取出processors字典中的key代表的元素
    p = Processor(sys.argv[2:])   #sys.argv[0]指.py程序本身,argv[2:]指从命令行获取的第二个参数

    print('start')
    p.start()