# -*- coding:utf-8 -*-

import random
import argparse
import os
from os import listdir
from os.path import join, isfile
import time
import json
import xml_helper
'''
设置trainval和test数据集包含的图片
'''


def gen_imagesets(vocpath=None):
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
    # write readme in vocpath
    readme_dict = dict()
    readme_dict['date'] = getTimeFlag()
    readme_dict['dataInfo'] = [vocpath.split('/')[-1]]
    readme_dict['author'] = "Ben"
    readme_dict['total_num'] = xml_numbers
    readme_dict['trainval_num'] = len(trainval_list)
    readme_dict['test_num'] = len(test_list)
    readme_file = os.path.join(vocpath,'readme.txt')
    with open(readme_file,'w') as f:
        json.dump(readme_dict, f, indent=4)
def getTimeFlag():
    return time.strftime("%Y-%m-%d-%H", time.localtime())




def statisticBboxInfo_Fun(imagelistFile=None,xmlFileBasePath=None):
    """
      imagelistFile is file , per line is a image(xml) file 
        not include jpg or xml 
    """
    line_count = 0
    label_count_dict=dict()
    with open(imagelistFile,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line :
                continue
            line_count += 1
            xmlFile = os.path.join(xmlFileBasePath, imagelistFile+'.xml')
            object_list = xml_helper.parseXmlFile_countBboxClassNum(
                xmlFile=xmlFile)
            for i_object in object_list:
                label = i_object['name']
                if label in label_count_dict:
                    label_count_dict[label] = label_count_dict[label] + 1
                else:
                    label_count_dict[label] = 1
    print("*"*100)
    print("image count in %s is : %d" % (imagelistFile, line_count))
    for key in sorted(label_count_dict.keys()):
        print("%s : %d" % (key, label_count_dict[key]))
    pass

def main():
    imagelistFile = "/workspace/data/BK/terror-dataSet-Dir/TERROR-DETECT-V1.0/ImageSets/Main/trainval.txt"
    xmlFileBasePath = "/workspace/data/BK/terror-dataSet-Dir/TERROR-DETECT-V1.0/Annotations"
    statisticBboxInfo_Fun(imagelistFile=imagelistFile,
                          xmlFileBasePath=xmlFileBasePath)
if __name__ == '__main__':
    main()
