# -*- coding:utf-8 -*
import os
import sys
import json
import hashlib
import threading
import Queue
import time
import numpy as np

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
    # wgetImageFromUrl(urlFile=justUrlListFile, saveBasePath=imageSaveDir)
    wgetImageFromUrl_MulThread(
        urlFile=justUrlListFile, saveBasePath=imageSaveDir)

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


GLOBAL_LOCK = threading.Lock()
ERROR_NUMBER = 0
IMAGE_SAVE_PATH=None
THREAD_DOWNLOAD_COUNT = 20
def wgetImageFromUrl_MulThread(urlFile=None, saveBasePath=None):
    # globale vars initialization
    global IMAGE_SAVE_PATH
    inputFileOp = open(urlFile,'r')
    logErOp = open(urlFile+'-wget-error.log','w')
    IMAGE_SAVE_PATH = saveBasePath
    if not os.path.exists(IMAGE_SAVE_PATH):
        os.makedirs(IMAGE_SAVE_PATH)
    queue = Queue.Queue(0)
    thread_prod = prod_worker(queue, inputFileOp)
    thread_prod.start()
    print('thread:', thread_prod.name, 'successfully started')
    time.sleep(10)
    for i in xrange(THREAD_DOWNLOAD_COUNT):
        exec('thread_cons_{} = cons_worker(queue,IMAGE_SAVE_PATH,logErOp)'.format(i))
        eval('thread_cons_{}.start()'.format(i))
    thread_prod.join()
    for i in xrange(THREAD_DOWNLOAD_COUNT):
        eval('thread_cons_{}.join()'.format(i))
    print('total error number:', ERROR_NUMBER)
    inputFileOp.close()
    logErOp.close()

    pass


class prod_worker(threading.Thread):
    """
    producing worker
    """
    global GLOBAL_LOCK

    def __init__(self, queue, infileOp):
        threading.Thread.__init__(self)
        self.queue = queue
        self.infileOp = infileOp
    def run(self):
        for line in self.infileOp:
            line = line.strip()
            if line == None or len(line) <=0:
                continue
            GLOBAL_LOCK.acquire()
            self.queue.put(line)
            GLOBAL_LOCK.release()
        GLOBAL_LOCK.acquire()
        print('thread:', self.name, 'successfully quit')
        GLOBAL_LOCK.release()


class cons_worker(threading.Thread):
    global GLOBAL_LOCK
    global IMAGE_SAVE_PATH
    def __init__(self, queue,savePath,logErOp):
        threading.Thread.__init__(self)
        self.queue = queue
        self.savePath = savePath
        self.logErOp = logErOp

    def download(self, url, output_path):
        err_flag = 0
        image_save_path = os.path.join(output_path,url.split('/')[-1])
        try:
            cmdStr = "wget %s -O %s -q" % (url, image_save_path)
            ret = os.system(cmdStr)
            count = 5
            while ret != 0 and count > 0:
                ret = os.system(cmdStr)
                count -= 1   
            if ret != 0:
                err_flag = 1
        except all as e:
            err_flag = 1
        return err_flag

    def run(self):
        global ERROR_NUMBER
        err_num = 0
        while(not self.queue.empty()):
            if GLOBAL_LOCK.acquire(False):
                # customized downloading code
                url = self.queue.get()
                GLOBAL_LOCK.release()
                err_flag = self.download(url, IMAGE_SAVE_PATH)
                if err_flag == 1:
                    # wget error
                    err_num += 1
                    GLOBAL_LOCK.acquire()
                    self.logErOp.write(url)
                    self.logErOp.flush()
                    GLOBAL_LOCK.release()
                    pass
            else:
                pass
        GLOBAL_LOCK.acquire()
        ERROR_NUMBER += err_num
        print('thread:', self.name, 'successfully quit')
        GLOBAL_LOCK.release()


def readImage_fun(isUrlFlag=None, imagePath=None):
    """
        isUrlFlag == True , then read image from url
        isUrlFlag == False , then read image from local path
    """
    im = None
    if isUrlFlag == True:
        try:
            data = urllib.urlopen(imagePath.strip()).read()
            nparr = np.fromstring(data, np.uint8)
            if nparr.shape[0] < 1:
                im = None
        except:
            im = None
        else:
            im = cv2.imdecode(nparr, 1)
        finally:
            return im
    else:
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if np.shape(im) == ():
        return None
    return im










































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

