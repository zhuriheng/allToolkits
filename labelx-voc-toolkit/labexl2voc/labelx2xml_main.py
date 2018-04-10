# -*- coding:utf-8 -*-
import os
import sys
import json
import argparse
import labelx2xml_helper

helpInfoStr = \
"""
    labelx convert to pascal voc xml file toolkit   
    actionFlag : 功能 flag
        1 : 将指定目录下的所有打标过的json 文件转换成 pascal xml 格式数据
            --labelxBasePath ,required
            --vocpath , optional
        2 : 将一个 pascal voc 数据集 添加到 另外一个数据集中
            --vocpath ,required 
            --finalVocpath , required
            将 vocpath 指向的数据集 添加到 finalVocpath 这个数据集中
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='labelx convert to pascal voc toolkit'
    )
    parser.add_argument('--actionFlag',
                        dest='actionFlag',
                        default=None,
                        required=True,
                        help='action flag int',
                        type=int)
    parser.add_argument('--vocpath',
                        dest='vocpath',
                        default=None,
                        help='vocpath for data generate',
                        type=str)
    parser.add_argument('--finalVocpath',
                        dest='finalVocpath',
                        default=None,
                        help='vocpath for data generate',
                        type=str)
    parser.add_argument('--labelxBasePath',
                        dest='labelxBasePath',
                        default=None,
                        help='labelx annotation file base path',
                        type=str)

    args = parser.parse_args()
    return args

args = parse_args()

def main():
    if args.actionFlag == None:
        print("WARNING %s" % (helpInfoStr))
    elif args.actionFlag == 1:
        labelxBasePath = args.labelxBasePath
        if labelxBasePath == None:
            print("labelxBasePath required")
            return -1
        labelxBasePath = os.path.abspath(labelxBasePath)
        vocpath = args.vocpath
        if vocpath == None:
            vocpath = labelxBasePath+'-vocResult'
        labelx2xml_helper.covertLabelxMulFilsToVoc_Fun(
            labelxPath=labelxBasePath, vocResultPath=vocpath)
        pass
    elif args.actionFlag == 2:
        vocpath = args.vocpath
        finalVocpath = args.finalVocpath
        if vocpath == None or finalVocpath == None:
            print("vocpath and finalVocpath is required")
            return -1
        res = labelx2xml_helper.mergePascalDataset(
            littlePath=vocpath, finalPath=finalVocpath)
        if res == 'error':
            return 1
        pass
    pass

if __name__ == '__main__':
    res = main()
    if res == -1:
        print(helpInfoStr)
    else:
        print("RUN SUCCESS")

