# -*- coding:utf-8 -*-
import os
import sys
import json
from lxml import etree
# from PIL import Image
import cv2
import labelxJson_helper
import image_helper
import numpy as np
# 定义一个常量字典
PASCAL_VOC={
    'folder': 'VOC2007',
    'source':{
        'database': 'TERROR-DETECT',
        'annotation': 'PASCAL VOC2007',
        'image': 'flickr',
        'flickrid': 'NULL'
    },
    'owner':{
        'flickrid':'NULL',
        'name':'QiNiu'
    },
    'segmented':'0',
    'imageChannel':'3',
    'object':{
        'pose': 'Unspecified',
        'truncated':'0',
        'difficult':'0'
    }
}

def pascalVocXmlTemplate():
    rootElememt = etree.Element('annotation')
    folder_Element_1 = etree.SubElement(rootElememt, 'folder')
    folder_Element_1.text = PASCAL_VOC['folder']
    filename_Element_1 = etree.SubElement(rootElememt, 'filename')
    source_Element_1 = etree.SubElement(rootElememt, 'source')
    owner_Element_1 = etree.SubElement(rootElememt, 'owner')
    size_Element_1 = etree.SubElement(rootElememt, 'size')
    segmented_Element_1 = etree.SubElement(rootElememt, 'segmented')
    segmented_Element_1.text = PASCAL_VOC['segmented']
    return rootElememt
def appendFileNameText(root=None, fileName=None):
    rootElememt = root
    rootElememt.find('filename').text = fileName
def appendSourceText(root=None):
    rootElememt = root
    source_Element_1 = rootElememt.find('source')
    database = etree.SubElement(source_Element_1, 'database')
    database.text= PASCAL_VOC['source']['database'] 
    annotation = etree.SubElement(source_Element_1, 'annotation')
    annotation.text = PASCAL_VOC['source']['annotation'] 
    image = etree.SubElement(source_Element_1, 'image')
    image.text = PASCAL_VOC['source']['image'] 
    flickrid = etree.SubElement(source_Element_1, 'flickrid')
    flickrid.text = PASCAL_VOC['source']['flickrid']
def appendOwnerText(root=None):
    rootElememt = root
    owner_Element_1 = rootElememt.find('owner')
    flickrid = etree.SubElement(owner_Element_1, 'flickrid')
    flickrid.text = PASCAL_VOC['owner']['flickrid']
    name = etree.SubElement(owner_Element_1, 'name')
    name.text = PASCAL_VOC['owner']['name']
def appendSizeText(root=None,imagePath=None):
    # 获取图片信息
    # img = Image.open(imagePath)
    img = None
    try:
        img = cv2.imread(imagePath)
    except :
        img = None
    if np.shape(img) == ():
        print("ERROR INFO : %s can't read"%(imagePath))
        return None
    img_height, img_width,img_depth = img.shape
    # img_width, img_height = img.size
    rootElememt = root
    size_Element_1 = rootElememt.find('size')
    width = etree.SubElement(size_Element_1, 'width')
    width.text = str(img_width)
    height = etree.SubElement(size_Element_1, 'height')
    height.text = str(img_height)
    depth = etree.SubElement(size_Element_1, 'depth')
    depth.text = str(img_depth)
    if img_depth != 3:
        print("ERROR info : %s channel is %d" % (imagePath, img_depth))
        return None
    return 0
def appendObject(root=None,objectDict=None):
    """
        objectDict:
        {
            "class": "guns_true_F", 
            "bbox": [[405, 112], [487, 112], [487, 180], [405, 180]], 
            "ground_truth": true
        }
    """
    _object = createObject(objectDict=objectDict)
    root.append(_object)
def createObject(objectDict=None):
    _object = etree.Element('object')
    _name = etree.SubElement(_object, 'name')
    _name.text = objectDict['class']
    _pose = etree.SubElement(_object, 'pose')
    _pose.text = PASCAL_VOC['object']['pose']
    _truncated = etree.SubElement(_object, 'truncated')
    _truncated.text = PASCAL_VOC['object']['truncated']
    _difficults = etree.SubElement(_object, 'difficult')
    _difficults.text = PASCAL_VOC['object']['difficult']
    _bndbox = etree.SubElement(_object, 'bndbox')
    _xmin = etree.SubElement(_bndbox, 'xmin')
    _xmin.text = convertToInt_str(objectDict['bbox'][0][0])
    _ymin = etree.SubElement(_bndbox, 'ymin')
    _ymin.text = convertToInt_str(objectDict['bbox'][0][1])
    _xmax = etree.SubElement(_bndbox, 'xmax')
    _xmax.text = convertToInt_str(objectDict['bbox'][2][0])
    _ymax = etree.SubElement(_bndbox, 'ymax')
    _ymax.text = convertToInt_str(objectDict['bbox'][2][1])
    return _object

def adjustBboxPosition(root=None):
    root = root  # annotation
    size = root.find('size')
    width_int = int(float(size.find('width').text))
    height_int = int(float(size.find('height').text))
    object_list = root.findall('object')
    for object in object_list:
        bndbox = object.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        if xmin <= 0:
            xmin = 1
        if ymin <= 0:
            ymin = 1
        if xmax >= width_int:
            xmax = width_int - 1
        if ymax >= height_int:
            ymax = height_int-1
        if xmin >= xmax:
            xmax = xmin + 1
        if ymin >= ymax:
            ymax = ymin + 1
        if xmax > width_int or ymax > height_int:
            # remove the object
            root.remove(object)
        else:
            bndbox.find('xmin').text = convertToInt_str(xmin)
            bndbox.find('ymin').text = convertToInt_str(ymin)
            bndbox.find('xmax').text = convertToInt_str(xmax)
            bndbox.find('ymax').text = convertToInt_str(ymax)
def convertToInt_str(input):
    output = None
    if type(input) == str:
        output = int(float(input))
    elif type(input) == float:
        output = int(str)
    elif type(input) == int:
        output = input
    return str(output)

def renameFileName(xmlFileName=None,newFileName=None):
    root = etree.parse(xmlFileName).getroot()
    imageName = newFileName.split('/')[-1].split('.')[0]+'.jpg'
    rootElememt = root
    rootElememt.find('filename').text = imageName
    writeXmlFile(root=rootElememt, xmlFileName=newFileName)


def createXmlFileByLabelXJsonList(labelxJsonLine=None, basePath=None):
    """
        这个函数的作用是：根据一个图片的labelx 标注信息，生成 pascal voc xml 文件
    """
    key, value = labelxJson_helper.get_jsonList_line_labelInfo(
        line=labelxJsonLine)
    if value == None:
        print("WARNING: %s don't contains labeled info" % (labelxJsonLine))
    # xml file template
    root = pascalVocXmlTemplate()
    imageName = key.split('/')[-1]
    appendFileNameText(root=root, fileName=imageName)
    appendSourceText(root=root)
    appendOwnerText(root=root)
    imageLocalImagePath = os.path.join(basePath,'JPEGImages', imageName)
    res = appendSizeText(root=root, imagePath=imageLocalImagePath)
    if res == None:
        # because the image can't read ,so delete the image
        errorImageDir = os.path.join(basePath,'errorImage')
        if not os.path.exists(errorImageDir):
            os.makedirs(errorImageDir)
        errorImage_path_name = os.path.join(errorImageDir, imageLocalImagePath.split('/')[-1])
        cmdStr = "mv %s %s" % (imageLocalImagePath, errorImage_path_name)
        res = os.system(cmdStr)
        if res != 0:
            print("run command error %s" % (cmdStr))
            exit()
        return "error"
    for i_bbox in value:
        appendObject(root=root, objectDict=i_bbox)
    # adjust bbox position
    adjustBboxPosition(root=root)
    xmlBasePath = os.path.join(basePath, 'Annotations')
    if not os.path.exists(xmlBasePath):
        os.makedirs(xmlBasePath)
    xmlFileName = imageName.split('.')[0]+'.xml'
    saveXmlFilePath = os.path.join(xmlBasePath, xmlFileName)
    writeXmlFile(root = root, xmlFileName = saveXmlFilePath)
    return "success"
def writeXmlFile(root=None, xmlFileName=None):
    tree = etree.ElementTree(root)
    tree.write(xmlFileName, pretty_print=True)

def convertLabelxJsonListToXmlFile(jsonlistFile=None,datasetBasePath=None):
    with open(jsonlistFile,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            res = createXmlFileByLabelXJsonList(
                labelxJsonLine=line, basePath=datasetBasePath)
            if res == "error":
                print("ERROR")


def parseXmlFile_countBboxClassNum(xmlFile=None):
    tree = etree.parse(xmlFile)
    rooTElement = tree.getroot()
    object_list = []
    for child in rooTElement:
        if child.tag == "object":
            one_object_dict = {}
            one_object_dict['name'] = child.xpath('name')[0].text
            one_object_dict['xmin'] = child.xpath(
                'bndbox')[0].xpath('xmin')[0].text
            one_object_dict['ymin'] = child.xpath(
                'bndbox')[0].xpath('ymin')[0].text
            one_object_dict['xmax'] = child.xpath(
                'bndbox')[0].xpath('xmax')[0].text
            one_object_dict['ymax'] = child.xpath(
                'bndbox')[0].xpath('ymax')[0].text
            object_list.append(one_object_dict)
    return object_list



