# -*- coding:utf-8 -*
import os
import sys
import json
import hashlib
import threading
import Queue

def downloadImage_By_urllist(labelxjson=None, tempSaveDir=None, vocpath=None):
    imageSaveDir = os.path.join(vocpath, 'JPEGImages')
    if not os.path.exists(imageSaveDir):
        os.makedirs(imageSaveDir)
    imageSaveDir = os.path.abspath(imageSaveDir)
    justUrlListFile = os.path.join(tempSaveDir,'urlList.list')
    with open(labelxjson, 'r') as r_f, open(justUrlListFile,'w') as w_f:
        for line in r_f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            url = json.loads(line)['url']
            if url == None or len(url) <= 0:
                print("WARNING : %s url is None"%(line))
            else:
                w_f.write(url+'\n')
    wgetImageFromUrl(urlFile=justUrlListFile, saveBasePath=imageSaveDir)

# wget images
def wgetImageFromUrl(urlFile=None, saveBasePath=None):
    logFile = urlFile+'-wget.log'
    print("logFile : %s" % (logFile))
    logFile_error = urlFile+'-wget-error.log'
    print("logFile_error : %s" % (logFile_error))
    with open(urlFile, 'r') as r_f, open(logFile, 'w') as w_f, open(logFile_error, 'w') as w_e_f:
        for line in r_f.readlines():
            line = line.strip()
            if len(line) <= 0:
                continue
            savePath = os.path.join(saveBasePath, line.split('/')[-1])
            cmdStr = "wget %s -O %s -q" % (line, savePath)
            ret = os.system(cmdStr)
            count = 5
            while ret != 0 and count > 0:
                if os.path.exists(savePath):
                    os.remove(savePath)
                ret = os.system(cmdStr)
                count -= 1
            if ret == 0:
                w_f.write(line)
                w_f.write('\n')
                pass
            else:
                w_e_f.write(line)
                w_e_f.write('\n')
                pass
    pass


def wgetImageFromUrl_MulThread(urlFile=None, saveBasePath=None):
    # globale vars initialization
    GLOBAL_LOCK = threading.Lock()
    ERROR_NUMBER = 0
    FILE_NAME = str()
    
    pass



GLOBAL_LOCK = threading.Lock()
ERROR_NUMBER = 0
FILE_NAME = str()












































# not used ********************


def remove_same_image_by_md5():
    pass




def checkFileIsImags(filePath):
    if ('JPEG' in filePath.upper()) or ('JPG' in filePath.upper()) \
            or ('PNG' in filePath.upper()) or ('BMP' in filePath.upper()):
        return True
    return False
    pass

def getAllImages(basePath=None):
    allImageList=[]
    for parent,dirnames,filenames in os.walk(basePath):
        for file in filenames:
            imagePathName = os.path.join(parent,file)
            if checkFileIsImags(imagePathName):
                allImageList.append(imagePathName)
            else:
                # print("%s isn't image"%(imagePathName))
                pass
    return  allImageList
    pass

def md5_process(image=None):
    hash_md5 = hashlib.md5()
    with open(image,'rb') as f:
        for chunk in iter(lambda :f.read(4096),b""):
            hash_md5.update(chunk)
        return hash_md5.hexdigest()
    pass


def mv_same_md5_file(imageName=None,basePath=None):
    # mv image
    imageNameAndPath = os.path.join(basePath,'JPEGImages',imageName)
    imageSameSavePath = os.path.join(basePath,'JPEGImages-sameImage')
    if not os.path.exists(imageSameSavePath):
        os.makedirs(imageSameSavePath)
    cmdStrImg ="mv %s %s"%(imageNameAndPath,imageSameSavePath)
    # mv xml
    xmlFileName = imageName[:imageName.rfind('.')]+'.xml'
    xmlNameAndPath = os.path.join(basePath,'Annotations',xmlFileName)
    xmlSameSavePath = os.path.join(basePath,'Annotations-sameImage')
    if not os.path.exists(xmlSameSavePath):
        os.makedirs(xmlSameSavePath)
    cmdStrXml = "mv %s %s"%(xmlNameAndPath,xmlSameSavePath)
    img_result_flag = os.system(cmdStrImg)
    xml_result_flag = os.system(cmdStrXml)
    pass



def getNameByAbsolutePath(absolutePath=None):
    name = absolutePath.split('/')[-1]
    # name = name[:name.rfind('.')]
    return name
    pass

def md5_remove_model_fun(basePath=None):
    imagePath = os.path.join(basePath,'JPEGImages')
    xmlPath = os.path.join(basePath,'Annotations')
    allImagesPathList = getAllImages(basePath=imagePath)
    md5_imagaPath_dict = {}
    for imagePath in allImagesPathList:
        # imagePath is absolute path
        md5_key = md5_process(imagePath)
        if md5_key in md5_imagaPath_dict:
            print("%s --- %s same md5_key" % (getNameByAbsolutePath(absolutePath=imagePath),
                                              getNameByAbsolutePath(absolutePath=md5_imagaPath_dict.get(md5_key))))
            mv_same_md5_file(imageName=getNameByAbsolutePath(imagePath),basePath=basePath)
        else:
            md5_imagaPath_dict[md5_key] = imagePath
    print("before remove same image : %d"%(len(allImagesPathList)))
    print("after remove same image : %d"%(len(md5_imagaPath_dict)))
    pass

