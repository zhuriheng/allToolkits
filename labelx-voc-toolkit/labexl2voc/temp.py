# -*- coding:utf-8 -*-
import os
import sys
import json
from lxml import etree
"""
这个函数的作用是 临时的将
v0.9 中，原来对于一张图片 含有暴恐 和非暴恐 bbox 的，这些图片拿去从新打标9类了，
现在要做的是，对于这些图片，如果 v0.9 中的 not terror bbox 添加到 1.0 中
"""


def get_old_not_terror_object_list(xmlFile=None):
    tree = etree.parse(xmlFile)
    rooTElement = tree.getroot()
    object_list = []
    for child in rooTElement:
        if child.tag == "object":
            bbox_label = child.xpath('name')[0].text
            if bbox_label == "not terror":
                object_list.append(child)
    return object_list    
def addFun(odlxml=None,newxml=None):
    root = etree.parse(newxml).getroot()
    not_terror_object_list = get_old_not_terror_object_list(xmlFile=None)
    for new_object in not_terror_object_list:
        root.append(new_object)
    writeXmlFile(root=root, xmlFileName=newxml)
    pass


def writeXmlFile(root=None, xmlFileName=None):
    tree = etree.ElementTree(root)
    tree.write(xmlFileName, pretty_print=True)
def main():
    labeledFile_list = "/workspace/data/labelx-Dir/data/all-notInclude-notTerror-v0.9-File.list"
    xmlFileNewPath = '/workspace/data/BK/terror-dataSet-Dir/TERROR-DETECT-V1.0/Annotations'
    xmlFilePath_0_9 = "/workspace/data/BK/terror-dataSet-Dir/TERROR-DETECT-V0.9/Annotations"
    with open(labeledFile_list,'r') as f:
        file_list = [i.strip()+'.xml' for i in f.readlines() if i.strip()]
        for i in file_list:
            oldXml = os.path.join(xmlFilePath_0_9,i)
            if not os.path.isfile(oldXml):
                print("ERROR : %s" % (oldXml))
                return
            newXml = os.path.join(xmlFileNewPath, i)
            if not os.path.isfile(newXml):
                print("NEW  : %s don't exists" % (oldXml))
                continue
            addFun(odlxml=oldXml, newxml=newXml)
            
if __name__ == '__main__':
    main()
