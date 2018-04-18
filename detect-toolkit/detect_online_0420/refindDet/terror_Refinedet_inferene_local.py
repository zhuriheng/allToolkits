# -*- coding: utf-8 -*-

import os
import sys
import time
import traceback
import numpy as np
import terror_Refinedet_config as terror_Refinedet_config

from evals.utils import net_init_handler, net_inference_handler, CTX, \
    monitor_rt_load, monitor_rt_forward, monitor_rt_post

from evals.utils import *
from evals.utils.error import *
from evals.utils.header import Header, wrap_header
from evals.utils.logger import logger
from evals.utils.image import load_image
from evals.caffe_base.net import NetConfig, Net


@net_init_handler
def net_init(configs):

    CTX.logger.info("load configs: %s", configs)

    config = NetConfig()
    try:
        config.parse(configs)
    except (ErrorConfig, ErrorFileNotExist) as e:
        return None, str(e)
    except Exception, e:
        CTX.logger.error("caffe net create error: %s", traceback.format_exc())
        return None, str(e)

    net = Net()
    try:
        net.init(config)
    except Exception, e:
        CTX.logger.error("caffe net create error: %s", traceback.format_exc())
        return None, str(e)
    return {"net": net, "thresholds": configs['thresholds']}, ''


@net_inference_handler
def net_inference(model, reqs):

    net = model["net"]
    thresholds = model["thresholds"]
    CTX.logger.info("inference begin ...")

    try:
        pre_eval(net, reqs)
        output = eval(net, reqs)
        ret = post_eval(net, output, thresholds, reqs)
    except ErrorBase as e:
        return [{"code": e.code, "message": str(e)}], ''
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [{"code": 599, "message": str(e)}], ''

    return ret, ''


def preProcessImage(oriImage=None):
    img = cv2.resize(oriImage, (320, 320))
    img = img.astype(np.float32, copy=False)
    img = img - np.array([[[103.52, 116.28, 123.675]]])
    img = img * 0.017
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    return img


def pre_eval(net, reqs):
    '''
        prepare net forward data
        Parameters
        ----------
        net: net created by net_init
        reqs: parsed reqs from net_inference
        reqid: reqid from net_inference
        Return
        ----------
        code: error code, int
        message: error message, string
    '''
    cur_batchsize = len(reqs)
    if cur_batchsize > net.batch_size:
        for i in range(cur_batchsize):
            raise ErrorOutOfBatchSize(net.batch_size)
    net.images = []
    _t1 = time.time()
    for i in range(cur_batchsize):
        img = load_image(reqs[i]["data"]["uri"])
        height, width, _ = img.shape
        if height <= 32 or width <= 32:
            raise ErrorBase(400, "image too small " +
                            str(height) + "x" + str(width))
        if img.ndim != 3:
            raise ErrorBase(400, "image ndim is " +
                            str(img.ndim) + ", should be 3")
        net.images.append(preProcessImage(oriImage=img))

    _t2 = time.time()
    CTX.logger.info("load: %f\n", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)

    images = net.images
    net.imread(images)
    _t3 = time.time()
    CTX.logger.info("transform: %f\n", _t3 - _t2)
    monitor_rt_transform().observe(_t3 - _t2)


def post_eval(net, output, thresholds, reqs=None):
    '''
        parse net output, as numpy.mdarray, to EvalResponse
        Parameters
        ----------
        net: net created by net_init
        output: list of tuple(score, boxes)
        reqs: parsed reqs from net_inference
        reqid: reqid from net_inference
        Return
        ----------
        resps: list of EvalResponse{
            "code": <code|int>,
            "message": <error message|str>,
            "result": <eval result|object>,
            "result_file": <eval result file path|string>
        }
    '''
    resps = []
    # cur_batchsize = len(output['detection_out']) # len(output['detection_out'])  always 1
    cur_batchsize = len(reqs)
    _t1 = time.time()
    # output_bbox_list : bbox_count * 7
    output_bbox_list = output['detection_out'][0][0]
    image_result_dict = dict()  # image_id : bbox_list
    for i_bbox in output_bbox_list:
        image_id = int(i_bbox[0])
        image_data = net.images[image_id]
        w = image_data[1]
        h = image_data[0]
        if image_id >= cur_batchsize:
            break
        class_index = int(i_bbox[1])
        score = float(i_bbox[2])
        if class_index == 4 or class_index < 0:
            continue
        if score < thresholds[class_index]:
            continue
        if image_id in image_result_dict.keys():
            the_image_bbox_list = image_result_dict.get(image_id)
            pass
        else:
            the_image_bbox_list = []
            bbox_dict = dict()
        bbox_dict['cls'] = class_index
        bbox_dict['score'] = score
        bbox = i_bbox[3:7] * np.array([w, h, w, h])
        bbox = bbox.tolist()
        bbox_dict['bbox'] = bbox
        the_image_bbox_list.append(bbox_dict)
    resps = []
    for image_id in range(cur_batchsize):
        if image_id in image_result_dict.keys():
            res_list = image_result_dict.get(image_id)
        else:
            res_list = []
        result = {"detections": res_list}
        resps.append({"code": 0, "message": "", "result": result})
    _t2 = time.time()
    CTX.logger.info("post: %f\n", _t2 - _t1)
    monitor_rt_post().observe(_t2 - _t1)
    return resps


def eval(net, reqs):
    '''
        eval forward inference
        Return
        ---------
        output: network numpy.mdarray
    '''
    _t1 = time.time()
    output = net.forward()
    _t2 = time.time()
    CTX.logger.info("forward: %f\n", _t2 - _t1)
    monitor_rt_forward().observe(_t2 - _t1)

    CTX.logger.info('detection_out: {}'.format(output['detection_out']))

    if 'detection_out' not in output or len(output['detection_out']) < 1:
        raise ErrorForwardInference()
    return output
