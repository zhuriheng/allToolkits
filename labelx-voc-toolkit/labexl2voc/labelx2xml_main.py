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
    parser.add_argument('--labelxBasePath',
                        dest='labelxBasePath',
                        default=None,
                        required=True,
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
        labelxBasePath = os.path.abspath(labelxBasePath)
        vocpath = args.vocpath
        if vocpath == None:
            vocpath = labelxBasePath+'-vocResult'
        labelx2xml_helper.covertLabelxMulFilsToVoc_Fun(
            labelxPath=labelxBasePath, vocResultPath=vocpath)
        pass
    pass


if __name__ == '__main__':
    main()

"""
"""
