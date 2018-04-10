# -*- coding:utf-8 -*-

import random
import argparse
import os
from os import listdir
from os.path import join, isfile


__author__ = 'peic'

'''
设置trainval和test数据集包含的图片
'''


def gen_imagesets(vocpath):
    # ImageSets文件夹
    _IMAGE_SETS_PATH = join(vocpath, 'ImageSets')
    _MAin_PATH = join(vocpath, 'ImageSets/Main')
    _XML_FILE_PATH = join(vocpath, 'Annotations')

    # 创建ImageSets数据集
    if os.path.exists(_IMAGE_SETS_PATH):
        print('ImageSets dir is already exists')
        if os.path.exists(_MAin_PATH):
            print('Main dir is already in ImageSets')
        else:
            os.makedirs(_MAin_PATH)
    else:
        os.makedirs(_IMAGE_SETS_PATH)
        os.makedirs(_MAin_PATH)

    # 遍历XML文件夹
    xml_list = [x for x in listdir(_XML_FILE_PATH) if isfile(join(_XML_FILE_PATH, x)) and not x[0] == '.']
    random.shuffle(xml_list)
    xml_numbers = len(xml_list)
    test_percent, train_percent, val_percent = 0.07, 0.77, 0.16
    test_list = xml_list[:int(xml_numbers*test_percent)]
    train_list = xml_list[int(xml_numbers * test_percent):int(xml_numbers * (test_percent+train_percent))]
    val_list = xml_list[int(xml_numbers * (test_percent+train_percent)):]
    trainval_list = train_list + val_list

    r = '\n'.join([xml[:xml.rfind('.')] for xml in test_list])
    with open(os.path.join(_MAin_PATH, 'test.txt'), 'w+') as f:
        f.write(r)
        f.write('\n')

    r = '\n'.join([xml[:xml.rfind('.')] for xml in train_list])
    with open(os.path.join(_MAin_PATH, 'train.txt'), 'w+') as f:
        f.write(r)
        f.write('\n')

    r = '\n'.join([xml[:xml.rfind('.')] for xml in val_list])
    with open(os.path.join(_MAin_PATH, 'val.txt'), 'w+') as f:
        f.write(r)
        f.write('\n')

    r = '\n'.join([xml[:xml.rfind('.')] for xml in trainval_list])
    with open(os.path.join(_MAin_PATH, 'trainval.txt'), 'w+') as f:
        f.write(r)
        f.write('\n')
