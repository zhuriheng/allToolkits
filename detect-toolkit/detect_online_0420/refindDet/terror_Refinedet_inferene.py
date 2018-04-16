# -*- coding: utf-8 -*-

import os
import sys
import time
import traceback
import caffelog
import numpy as np

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

    if not os.path.isdir("./run/caffe"):
        os.mkdir("./run/caffe")
    caffelog.init_caffelog("./run/caffe/log")

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
    return {"net": net}, ''


@net_inference_handler
def net_inference(model, reqs):

    net = model["net"]
    CTX.logger.info("inference begin ...")

    try:
        pre_eval(net, reqs)
        output = eval(net, reqs)
        ret = post_eval(net, output, reqs)
    except ErrorBase as e:
        return [{"code": e.code, "message": str(e)}], ''
    except Exception as e:
        CTX.logger.error("inference error: %s", traceback.format_exc())
        return [{"code": 599, "message": str(e)}], ''

    return ret, ''


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
        net.images.append(img)

    _t2 = time.time()
    CTX.logger.info("load: %f\n", _t2 - _t1)
    monitor_rt_load().observe(_t2 - _t1)

    images = net.transform(net.images)
    net.imread(images)
    _t3 = time.time()
    CTX.logger.info("transform: %f\n", _t3 - _t2)
    monitor_rt_transform().observe(_t3 - _t2)


def post_eval(net, output, reqs=None):
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
    cur_batchsize = len(output['detection_out'])
    _t1 = time.time()
    for i in xrange(cur_batchsize):
        res_list = []
        img_id = output['detection_out'][i][0, :, 0].astype(np.int32)
        if len(img_id) == 0:
            CTX.logger.info("no face detected\n")
            result = {"detections": res_list}
            resps.append({"code": 0, "message": "", "result": result})
            continue
        clns = output['detection_out'][i][0, :, 1]
        scores = output['detection_out'][i][0, :, 2]
        h = net.images[img_id[0]].shape[0]
        w = net.images[img_id[0]].shape[1]
        boxes = output['detection_out'][i][0, :, 3:7] * np.array([w, h, w, h])
        boxes = boxes.astype(np.int32)
        # cluster detections for each class
        bnum = len(boxes)
        for j in xrange(bnum):
            det = {}
            if int(clns[j]) < 0:
                continue
            det['index'] = int(clns[j])  # j
            det['class'] = "face"  # class name始终为face
            det['score'] = float(scores[j])
            det['pts'] = []
            det['pts'].append([int(boxes[j][0]), int(boxes[j][1])])
            det['pts'].append([int(boxes[j][2]), int(boxes[j][1])])
            det['pts'].append([int(boxes[j][2]), int(boxes[j][3])])
            det['pts'].append([int(boxes[j][0]), int(boxes[j][3])])
            res_list.append(det)
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
