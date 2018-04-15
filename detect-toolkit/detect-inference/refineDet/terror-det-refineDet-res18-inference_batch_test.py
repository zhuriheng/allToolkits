#coding=utf-8
import numpy as np
import sys
import os
import cv2
sys.path.insert(0, "/workspace/data/BK/refineDet-Dir/RefineDet/python")
import caffe
from argparse import ArgumentParser
import time
import json
import urllib
from google.protobuf import text_format
from caffe.proto import caffe_pb2
"""
    bk terror caffe detect model , refineDet res 18 model
"""


def parser_args():
    parser = ArgumentParser('bk detect caffe  refineDet model!')
    # localImageNameFile & localImageBasePath
    parser.add_argument('--localImageNameFile', dest='localImageNameFile', help='local image name file',
                        default=None, type=str)
    parser.add_argument('--localImageBasePath', dest='localImageBasePath', help='local image base path',
                        default=None, type=str)
    # url list file name
    parser.add_argument('--urlfileName', dest='urlfileName', help='url file name',
                        default=None, type=str)
    parser.add_argument('--urlfileName_beginIndex', dest='urlfileName_beginIndex', help='begin index in the url file name',
                        default=0, type=int)
    parser.add_argument('--gpu', dest='gpu_id', help='The GPU ide to be used',
                        default=0, type=int)

    parser.add_argument('--modelBasePath', required=True, dest='modelBasePath', help='Path to the model',
                        default=None, type=str)
    parser.add_argument('--modelName', required=True, dest='modelName', help='the name of the model',
                        default=None, type=str)
    parser.add_argument('--deployFileName', required=True, dest='deployFileName', help='deploy file name',
                        default=None, type=str)
    parser.add_argument('--labelFileName', required=True, dest='labelFileName', help='label file name',
                        default=None, type=str)

    parser.add_argument('--threshold', dest='threshold',
                        default=0.1, type=float)
    return parser.parse_args()


"""
    temp change , as you need
"""
crop_size = 225
people_result_file_OP = None
terror_save_log = None


def temp_init():
    global people_result_file_OP
    global terror_save_log
    global crop_size
    crop_size = 225
    if args.urlfileName:
        people_result_file = args.urlfileName+'-' + \
            str(args.urlfileName_beginIndex)+"-result-batch.log"
        terror_save_log = args.urlfileName+'-TerrorDir'
    elif args.localImageNameFile:
        people_result_file = args.localImageNameFile+"-result.log"
        terror_save_log = args.localImageNameFile + '-TerrorDir'
    people_result_file_OP = open(people_result_file, 'w')

    pass


"""
    different setting
"""


def postProcess(output=None, imagePath=None, img_shape=None):
    """
        postprocess net inference result
        img_shape = [height,width]
    """
    w = img_shape[1]
    h = img_shape[0]
    bbox = output[0, :, 3:7] * np.array([w, h, w, h])
    cls = output[0, :, 1]
    conf = output[0, :, 2]
    result_dict = dict()
    result_dict['bbox'] = bbox.tolist()
    result_dict['cls'] = cls.tolist()
    result_dict['conf'] = conf.tolist()
    result_dict['imagePath'] = imagePath
    people_result_file_OP.write(json.dumps(result_dict)+'\n')


def postProcess_batch(output_batch=None,imagePath_batch=None, img_shape_batch=None):
    image_result_bath_dict = dict()
    print('*'*100)
    print(output_batch.shape)
    output_bbox = output_batch[0][0]
    for i_bbox in output_bbox:
        image_index = i_bbox[0]
        image_index = int(image_index)
        w = img_shape_batch[image_index][1]
        h = img_shape_batch[image_index][0]
        if image_index in image_result_bath_dict.keys():
            image_result = image_result_bath_dict.get(image_index)
            bbox = i_bbox[3:7] * np.array([w, h, w, h])
            bbox = bbox.tolist()
            image_result['bbox'].append(bbox)
            conf = i_bbox[2]
            conf = float(conf)
            image_result['conf'].append(conf)
            cls = i_bbox[1]
            cls = float(cls)
            image_result['cls'].append(cls)
            pass
        else:
            image_result = dict()
            image_result['imagePath'] = imagePath_batch[image_index]
            image_reult_bbox_list = list()  
            bbox = i_bbox[3:7] * np.array([w, h, w, h])
            bbox = bbox.tolist()
            image_reult_bbox_list.append(bbox)
            image_result['bbox'] = image_reult_bbox_list
            image_reult_conf_list = list()
            conf = i_bbox[2]
            conf = float(conf)
            image_reult_conf_list.append(conf)
            image_result['conf'] = image_reult_conf_list
            image_reult_cls_list = list()
            cls = i_bbox[1]
            cls = float(cls)
            image_reult_cls_list.append(cls)
            image_result['cls'] = image_reult_cls_list
            image_result_bath_dict[image_index] = image_result
    for i in sorted(image_result_bath_dict.keys()):
        image_result = image_result_bath_dict.get(i)
        people_result_file_OP.write(json.dumps(image_result)+'\n')
    pass



def readImage_fun(isUrlFlag=False, imagePath=None):
    im = None
    if isUrlFlag:
        try:
            data = urllib.urlopen(imagePath.strip()).read()
            nparr = np.fromstring(data, np.uint8)
            if nparr.shape[0] < 1:
                im = None
        except:
            # print("Read Url Exception : %s" % (imagePath))
            im = None
        else:
            im = cv2.imdecode(nparr, 1)
        finally:
            return im
    else:
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if np.shape(im) == ():
        # print("waringing info : %s can't be read" % (imagePath))
        return None
    return im


def init_models():
    global CLS_LABEL_LIST
    global THRESHOLD
    global URLFLAG
    global IMAGE_SIZE
    global ONE_BATCH_SIZE
    ONE_BATCH_SIZE = 20
    URLFLAG = True if args.urlfileName else False
    THRESHOLD = args.threshold
    IMAGE_SIZE = 320
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    deployName = os.path.join(args.modelBasePath, args.deployFileName)
    modelName = os.path.join(args.modelBasePath, args.modelName)
    labelName = os.path.join(args.modelBasePath, args.labelFileName)
    net_cls = caffe.Net(deployName, modelName, caffe.TEST)
    with open(labelName, 'r') as f:
        CLS_LABEL_LIST = caffe_pb2.LabelMap()
        text_format.Merge(str(f.read()), CLS_LABEL_LIST)
    return net_cls


def getImagePath():
    imageList = []
    if args.urlfileName:
        with open(args.urlfileName, 'r') as f:
            for line in f.readlines()[args.urlfileName_beginIndex:]:
                line = line.strip()
                if line <= 0:
                    continue
                imageList.append(line)
    elif args.localImageNameFile:
        prefiex = "" if not args.localImageBasePath else args.localImageBasePath
        with open(args.localImageNameFile, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line <= 0:
                    continue
                line = os.path.join(prefiex, line)
                imageList.append(line)
    return imageList


def preProcess(oriImage=None):
    img = cv2.resize(oriImage, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32, copy=False)
    img = img - np.array([[[103.52, 116.28, 123.675]]])
    img = img * 0.017
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    return img

    pass


def infereneAllImage(net_cls=None, imageList=None, urlFlag=None):
    batch_image_data = []
    batch_image_path = []
    batch_image_h_w = []
    batch_count = 0
    for image_path in imageList:
        oriImg = readImage_fun(isUrlFlag=urlFlag, imagePath=image_path)
        if np.shape(oriImg) == ():
            print("ReadImageError : %s" % (image_path))
            continue
        img = preProcess(oriImage=oriImg)
        batch_image_data.append(img)
        batch_image_path.append(image_path)
        h_w = [oriImg.shape[0], oriImg.shape[1]]
        batch_image_h_w.append(h_w)
        batch_count += 1
        if batch_count == ONE_BATCH_SIZE:
            net_cls.blobs['data'].data[...] = batch_image_data
            output = net_cls.forward()
            detection_out_batch_result = output['detection_out']
            postProcess_batch(output_batch = detection_out_batch_result, imagePath_batch = batch_image_path, img_shape_batch = batch_image_h_w)
            batch_image_data = []
            batch_image_path = []
            batch_image_h_w = []
            batch_count = 0
    # process not enough 20
    last_batch_size = len(batch_image_data)
    if last_batch_size == 0:
        return
    else:
        in_shape = net_cls.blobs['data'].data.shape
        in_shape = (last_batch_size, in_shape[1], in_shape[2], in_shape[3])
        net_cls.blobs['data'].reshape(*in_shape)
        net_cls.reshape()
        net_cls.blobs['data'].data[...] = batch_image_data
        output = net_cls.forward()
        detection_out_batch_result = output['detection_out']
        postProcess_batch(output_batch=detection_out_batch_result,
                          imagePath_batch=batch_image_path, img_shape_batch=batch_image_h_w)
    return


args = parser_args()
CLS_LABEL_LIST = None
THRESHOLD = None
URLFLAG = None
IMAGE_SIZE = None
ONE_BATCH_SIZE = None


def main():
    temp_init()  # serve different need
    net_cls = init_models()
    imageList = getImagePath()
    infereneAllImage(net_cls=net_cls, imageList=imageList, urlFlag=URLFLAG)


if __name__ == '__main__':
    main()


"""
python terror-det-refineDet-res18-inference.py \
--urlfileName ./data/test_url.list \
--modelBasePath ./models/ \
--modelName terror-res18-320x320-t2_iter_130000.caffemodel \
--deployFileName deploy.prototxt \
--labelFileName  labelmap_bk.prototxt  \
--gpu 0

"""
