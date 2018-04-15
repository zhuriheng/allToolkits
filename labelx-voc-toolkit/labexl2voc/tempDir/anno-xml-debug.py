# -*- coding:utf-8 -*-
import os
import sys
import json
import argparse
from lxml import etree

def parse_args():
    parser = argparse.ArgumentParser(
        description='process terror detect train data convert annotation xml file')
    parser.add_argument('--inputBasePath',
                        dest='inputBasePath', default=None, type=str)
    parser.add_argument('--outputBasePath',
                        dest='outputBasePath', default=None, type=str)
    return parser.parse_args()


def convertXml_Fun(inpuXml=None):
    input_xml_absolut_path = os.path.join(args.inputBasePath, inpuXml)
    output_xml_absolut_path = os.path.join(args.outputBasePath, inpuXml)
    print(input_xml_absolut_path)
    doc = etree.parse(input_xml_absolut_path)
    root = doc.getroot() # annotation
    size = root.find('size')
    width_int = int(float(size.find('width').text))
    height_int=int(float(size.find('height').text))
    object_list = root.findall('object')
    for object in object_list:
        bndbox = object.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        if xmin <=0:
            xmin = 1
        if ymin <=0:
            ymin = 1
        if xmax >= width_int:
            xmax = width_int -1
        if ymax >= height_int:  
            ymax = height_int-1
        if xmin >= xmax:
            xmax = xmin + 1
        if ymin >= ymax:
            ymax = ymin + 1
        if xmax > width_int or ymax > height_int:
            # remove the object
            print(xmin)
            print(ymin)
            print(xmax)
            print(ymax)
            root.remove(object)
        else:
            bndbox.find('xmin').text = str(xmin)
            bndbox.find('ymin').text = str(ymin)
            bndbox.find('xmax').text = str(xmax)
            bndbox.find('ymax').text = str(ymax)
    doc.write(output_xml_absolut_path)
    pass

args = parse_args()
if __name__ == '__main__':
    if not os.path.exists(args.outputBasePath):
        os.makedirs(args.outputBasePath)
    for xmlFile in os.listdir(args.inputBasePath):
        convertXml_Fun(inpuXml=xmlFile)
        pass
    pass

