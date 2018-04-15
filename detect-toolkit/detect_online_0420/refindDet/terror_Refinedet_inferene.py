# -*- coding:utf-8 -*-
import numpy as np
import sys
import os
import cv2
import caffe

from terror_Refinedet_config import Config

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
    monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image


def init_models(use_divice, net_def_file, model_file):
    if use_divice == "GPU":
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    # initialize model
    net = caffe.Net(net_def_file, model_file, caffe.TEST)
    return net

@create_net_handler
def create_net(configs):

    CTX.logger.info("load configs: %s", configs)
    net = init_models(configs['use_device'].upper(),
                                configs['model_files']['deploy.prototxt'],
                                configs['model_files']['weight.caffemodel'])
    return {"net": net, "batch_size": configs['batch_size']}, 0, ''


def preProcess(oriImage=None):
    img = cv2.resize(oriImage, (320, 320))
    img = img.astype(np.float32, copy=False)
    img = img - np.array([[[103.52, 116.28, 123.675]]])
    img = img * 0.017
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    return img


def net_preprocess_Fun(req=None):
    
    pass

@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    text_detector = model['text_detector']
    batch_size = model['batch_size']
    CTX.logger.info("inference begin ...")

    try:
        imges_with_type = pre_eval(batch_size, reqs)
        output = eval(text_detector, imges_with_type)
        ret = post_eval(text_detector, output, reqs)

    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)

    return ret, 0, ''


"""
    terror refinedet res18 model inference
"""

def parser_args():
    parser = ArgumentParser('terror refinedet res18 model inference')
    # image list file
    parser.add_argument('--imageNameList', dest='imagePathList', help='image name or url list file ',
                        default=None, type=str)
    # local path or url path flag
    parser.add_argument('--imageFlag', dest='imageFlag', help='imageFlag : local or url',
                        default=None, type=str)
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


def postProcess_batch(output_batch=None, imagePath_batch=None, img_shape_batch=None):
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


@create_net_handler
def create_net(configs):

    CTX.logger.info("load configs: %s", configs)
    text_detector = init_models(configs['use_device'].upper(),
                                configs['model_files']['deploy.prototxt'],
                                configs['model_files']['weight.caffemodel'])
    return {"text_detector": text_detector, "batch_size": configs['batch_size']}, 0, ''

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
            postProcess_batch(output_batch=detection_out_batch_result,
                              imagePath_batch=batch_image_path, img_shape_batch=batch_image_h_w)
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

import pyximport
pyximport.install()
import argparse
import cv2
import caffe
from src.cfg import Config
from src.other import draw_boxes, resize_im, refine_boxes, calc_area_ratio, CaffeModel
from src.detectors import TextProposalDetector, TextDetector
from src.utils.timer import Timer
import json
import time
import traceback

from evals.utils import create_net_handler, net_preprocess_handler, net_inference_handler, CTX, \
    monitor_rt_load, monitor_rt_forward, monitor_rt_post
from evals.utils.error import *
from evals.utils.image import load_image


def init_models(use_divice, net_def_file, model_file):
    if use_divice == "GPU":
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # initialize the detectors
    text_proposals_detector = TextProposalDetector(
        CaffeModel(net_def_file, model_file))
    text_detector = TextDetector(text_proposals_detector)

    return text_detector


def text_detect(text_detector, im, img_type):
    if img_type == "others":
        return [], 0

    im_small, f = resize_im(im, Config.SCALE, Config.MAX_SCALE)

    timer = Timer()
    timer.tic()
    text_lines = text_detector.detect(im_small)
    text_lines = text_lines / f  # project back to size of original image
    text_lines = refine_boxes(im, text_lines, expand_pixel_len=Config.DILATE_PIXEL,
                              pixel_blank=Config.BREATH_PIXEL, binary_thresh=Config.BINARY_THRESH)
    text_area_ratio = calc_area_ratio(text_lines, im.shape)
    print "Number of the detected text lines: %s" % len(text_lines)
    print "Detection Time: %f" % timer.toc()
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if Config.DEBUG_SAVE_BOX_IMG:
        im_with_text_lines = draw_boxes(
            im, text_lines, is_display=False, caption=image_path, wait=False)
        if im_with_text_lines is not None:
            cv2.imwrite(image_path+'_boxes.jpg', im_with_text_lines)

    return text_lines, text_area_ratio


def dump_result(text_lines, text_area_ratio, img_type):
    text_detect_result = dict()
    text_detect_result['bboxes'] = text_lines
    text_detect_result['area_ratio'] = text_area_ratio
    text_detect_result['img_type'] = img_type

    return text_detect_result


@create_net_handler
def create_net(configs):

    CTX.logger.info("load configs: %s", configs)
    text_detector = init_models(configs['use_device'].upper(),
                                configs['model_files']['deploy.prototxt'],
                                configs['model_files']['weight.caffemodel'])
    return {"text_detector": text_detector, "batch_size": configs['batch_size']}, 0, ''


@net_preprocess_handler
def net_preprocess(model, req):
    CTX.logger.info("PreProcess...")
    return req, 0, ''


@net_inference_handler
def net_inference(model, reqs):
    text_detector = model['text_detector']
    batch_size = model['batch_size']
    CTX.logger.info("inference begin ...")

    try:
        imges_with_type = pre_eval(batch_size, reqs)
        output = eval(text_detector, imges_with_type)
        ret = post_eval(text_detector, output, reqs)

    except ErrorBase as e:
        return [], e.code, str(e)
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [], 599, str(e)

    return ret, 0, ''


def pre_eval(batch_size, reqs):

    cur_batchsize = len(reqs)
    if cur_batchsize > batch_size:
        raise ErrorOutOfBatchSize(batch_size)

    ret = []
    _t1 = time.time()
    for i in range(cur_batchsize):
        img = load_image(reqs[i]["data"]["uri"], body=reqs[i]['data']['body'])
        img_type = reqs[i]["params"]["image_type"]
        ret.append((img, img_type))

    _t2 = time.time()
    CTX.logger.info("load: %f", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)

    return ret


def post_eval(text_detector, output, reqs=None):

    resps = []
    cur_batchsize = len(output)
    _t1 = time.time()
    for i in xrange(cur_batchsize):
        text_bboxes = output[i][0]
        text_area_ratio = output[i][1]
        img_type = output[i][2]
        res_list = []
        if len(text_bboxes) == 0:
            CTX.logger.info("no text detected")
            resps.append({"code": 0, "message": "", "result": {}})
            continue
        result = dump_result(text_bboxes, text_area_ratio, img_type)
        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


def eval(text_detector, imges_with_type):
    output = []
    _t1 = time.time()
    for i in range(len(imges_with_type)):
        text_bboxes, text_area_ratio = text_detect(
            text_detector, imges_with_type[i][0], imges_with_type[i][1])
        output.append((text_bboxes, text_area_ratio, imges_with_type[i][1]))
    _t2 = time.time()
    CTX.logger.info("forward: %f", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)
    return output
