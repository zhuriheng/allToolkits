# -*- coding:utf-8 -*-
import os
import sys
import json
import time
import  gen_imagesets
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
    gen_imagesets.gen_imagesets(vocpath=vocpath)
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


def getFileCountInDir(dirPath=None):
    file_list = [i for i in os.listdir(dirPath) if i[0] != '.' and os.path.isfile(os.path.join(dirPath, i))]
    return [len(file_list), sorted(file_list)]


def mergePascalDataset(littlePath=None, finalPath=None):
    if not os.path.exists(finalPath):
        os.makedirs(finalPath)
    if not os.path.exists(os.path.join(finalPath, 'JPEGImages')):
        os.makedirs(os.path.join(finalPath, 'JPEGImages'))
    if not os.path.exists(os.path.join(finalPath, 'Annotations')):
        os.makedirs(os.path.join(finalPath, 'Annotations'))
    if not os.path.exists(os.path.join(finalPath, 'ImageSets', 'Main')):
        os.makedirs(os.path.join(finalPath, 'ImageSets', 'Main'))
    # merge image and merge xml
    littlePath_image = os.path.join(littlePath, 'JPEGImages')
    finalPath_image = os.path.join(finalPath, 'JPEGImages')
    littlePath_image_count = getFileCountInDir(littlePath_image)[0]
    littlePath_xml = os.path.join(littlePath, 'Annotations')
    finalPath_xml = os.path.join(finalPath, 'Annotations')
    littlePath_xml_count = getFileCountInDir(littlePath_xml)[0]
    if littlePath_image_count != littlePath_xml_count:
        print("ERROR : %s JPEGImages-nums unequals Annotations-nums" %(littlePath))
        return "error"
    # cmdStr_cp_image = "cp %s/* %s" % (littlePath_image, finalPath_image)
    cmdStr_cp_image = "for i in `ls %s`;do cp \"%s/\"$i %s;done;" % (
        littlePath_image, littlePath_image, finalPath_image)
    # cmdStr_cp_xml = "cp %s/* %s" % (littlePath_xml, finalPath_xml)
    cmdStr_cp_xml = "for i in `ls %s`;do cp \"%s/\"$i %s;done;" % (
        littlePath_xml, littlePath_xml, finalPath_xml)
    res = os.system(cmdStr_cp_image)
    if res != 0:
        print("ERROR : %s" % (cmdStr_cp_image))
        return "error"
    else:
        print("SUCCESS : %s" % (cmdStr_cp_image))
    res = os.system(cmdStr_cp_xml)
    if res != 0:
        print("ERROR : %s" % (cmdStr_cp_xml))
        return "error"
    else:
        print("SUCCESS : %s" % (cmdStr_cp_xml))
    # merge txt file
    littlePath_main = os.path.join(littlePath, 'ImageSets', 'Main')
    finalPath_main = os.path.join(finalPath, 'ImageSets', 'Main')
    textFile_list = getFileCountInDir(dirPath=littlePath_main)[1]
    for i in textFile_list:
        little_file = os.path.join(littlePath_main,i)
        final_file = os.path.join(finalPath_main, i)
        cmdStr = "cat %s >> %s" % (little_file, final_file)
        res = os.system(cmdStr)
        if res != 0:
            print("ERROR : %s" % (cmdStr))
            return 'error'
    # recode log
    record_log_file = os.path.join(finalPath,'update_log.log')
    with open(record_log_file,'a') as f:
        f.write("update info : %s add dataset ::: %s\n" % (getTimeFlag(), littlePath.split('/')[-1]))
        littlePath_readme = os.path.join(littlePath, 'readme.txt')
        littlePath_readme_dict = json.load(open(littlePath_readme,'r'))
        f.write(json.dumps(littlePath_readme_dict)+'\n')
    little_readme_file = os.path.join(littlePath, 'readme.txt')
    little_readme_file_dict = json.load(open(little_readme_file, 'r'))
    final_readme_file = os.path.join(finalPath, 'readme.txt')
    if not os.path.exists(final_readme_file):
        cmdStr = "cp %s %s" % (little_readme_file, final_readme_file)
        res = os.system(cmdStr)
        if res != 0:
            return 'error'
    else:
        final_readme_file_dict = json.load(open(final_readme_file, 'r'))
        for key in final_readme_file_dict.keys():
            if key == "date":
                final_readme_file_dict[key] = getTimeFlag()
            if key == "dataInfo":
                for i in little_readme_file_dict[key]:
                    final_readme_file_dict[key].append(i)
            if key == "author":
                final_readme_file_dict[key] = 'Ben'
            elif key in ['total_num', 'trainval_num', 'test_num']:
                final_readme_file_dict[key] = final_readme_file_dict[key] + \
                    little_readme_file_dict[key]
        with open(final_readme_file,'w') as f:
            json.dump(final_readme_file_dict, f, indent=4)
    pass


def getTimeFlag():
    return time.strftime("%Y-%m-%d-%H", time.localtime())

def gen_imageset_Fun(vocPath=None):
    gen_imagesets.gen_imagesets(vocpath=vocPath)
    pass


def statisticBboxInfo_Fun(vocPath=None):
    pass
