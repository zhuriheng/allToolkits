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
            --flag ,"overwrite,append"  
                    overwrite 表示如果 存在 xml文件名相同，那么就覆盖原有的xml文件，重新生成。
                    append 表示如果 存在xml 文件，那么进行 bbox 追加
            将 vocpath 指向的数据集 添加到 finalVocpath 这个数据集中
        3 : 根据已经有的图片和xmL文件生成 ImageSets/Main，readme.txt
            --vocpath ,required 
        4 : 统计vopath bbox 的类别信息
            --vocpath ,required 
        5 : 抽样画图，抽样画 pascal voc 格式的数据
            --vocpath ,required
            会 将画的图 保存在 vocpath+'-draw' 目录下。
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
    elif args.actionFlag == 3:
        vocpath = args.vocpath
        if vocpath == None:
            print("vocpath is required")
            return -1
        labelx2xml_helper.gen_imageset_Fun(vocPath=vocpath)
        pass
    elif args.actionFlag == 4:
        vocpath = args.vocpath
        if vocpath == None:
            print("vocpath is required")
            return -1
        labelx2xml_helper.statisticBboxInfo_Fun(vocPath=vocpath)
        pass
    elif args.actionFlag == 5:
        vocpath = args.vocpath
        if vocpath == None:
            print("vocpath is required")
            return -1
        labelx2xml_helper.drawImageWithBbosFun(vocPath=vocpath)
    pass

if __name__ == '__main__':
    res = main()
    if res == -1:
        print(helpInfoStr)
    else:
        print("RUN SUCCESS")

