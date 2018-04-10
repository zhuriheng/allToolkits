# -*- coding:utf-8 -*-
import os
import sys
import json
from gen_imagesets import gen_imagesets
import image_helper
import xml_helper
import labelxJson_helper


def process_labelx_jsonFile_Fun(json_file_absolutePath=None, tempSaveDir=None, vocpath=None):
    # 下载 对应的image,保存下载的图片到 vocpath+'/JPEGImages'
    image_helper.downloadImage_By_urllist(labelxjson=json_file_absolutePath, tempSaveDir=tempSaveDir, vocpath=vocpath)
    # 将 labelx 标注的数据 转换 pascal voc xml 文件
    xml_helper.convertLabelxJsonListToXmlFile(
        jsonlistFile=json_file_absolutePath, datasetBasePath=vocpath)
    # 这个是生成 pascal voc 格式的数据集 xml  jpg txt
    gen_imagesets(vocpath=vocpath)
    pass

def covertLabelxMulFilsToVoc_Fun(labelxPath=None,vocResultPath=None):
    inputDir = labelxPath
    tempSaveDir = labelxPath+"-xmlNeedTempFileDir"
    vocpath = vocResultPath
    if not os.path.exists(tempSaveDir):
        os.makedirs(tempSaveDir)
    if not os.path.exists(vocpath):
        os.makedirs(vocpath)
    # 1 : mergeAllJsonListFileToOneFile 将多个jsonlist 合并成一个，并按照url 去重
    finalOneFile = labelxJson_helper.mergeAllJsonListFileToOneFile(
        inputDir=inputDir, tempSaveDir=tempSaveDir)
    # 2 : 根据整合生成的一个总文件，开始下载图片，生成 xml 文件
    process_labelx_jsonFile_Fun(
        json_file_absolutePath=finalOneFile, tempSaveDir=tempSaveDir, vocpath=vocpath)
    pass


def mergePascalDataset(littlePath=None, finalPath=None):

    # merge image
    littlePath_image = 
    # merge xml
    # merge txt file
    # recode log
    pass
