from __future__ import print_function

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
sys.path.insert(0, os.path.join('/opt/dcn', 'rfcn'))
import rfcn._init_paths
import random
import cv2
import json
import traceback
import mxnet as mx
import numpy as np
import copy
import urllib
from rfcn.symbols import *
from rfcn.config.config import config, update_config
from core.tester import Predictor, im_detect, im_proposal, vis_all_detection, draw_all_detection
from lib.utils.load_model import load_param
from lib.utils.image import resize, transform
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from nms.box_voting import py_box_voting_wrapper

from evals.utils import _make_synset, infer_output_marshal
from evals.utils.image import load_image
from evals.utils.error import ErrorBase
from evals.mxnet_base import net
import time


def get_net(cfg, ctx, prefix, epoch, has_rpn):
    try:
        if has_rpn:
            sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
            sym = sym_instance.get_symbol(cfg, is_train=False)
        else:
            sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
            sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)

        # load model
        arg_params, aux_params = load_param(prefix, epoch, process=True)

        # infer shape
        SHORT_SIDE = config.SCALES[0][0]
        LONG_SIDE = config.SCALES[0][1]
        DATA_NAMES = ['data', 'im_info']
        LABEL_NAMES = None
        DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)),
                       ('im_info', (1, 3))]
        LABEL_SHAPES = None
        data_shape_dict = dict(DATA_SHAPES)
        sym_instance.infer_shape(data_shape_dict)
        sym_instance.check_parameter_shapes(
            arg_params, aux_params, data_shape_dict, is_train=False)

        # decide maximum shape
        max_data_shape = [
            [('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
        if not has_rpn:
            max_data_shape.append(
                ('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

        # create predictor
        predictor = Predictor(sym, DATA_NAMES, LABEL_NAMES,
                              context=ctx, max_data_shapes=max_data_shape,
                              provide_data=[DATA_SHAPES], provide_label=[
                                  LABEL_SHAPES],
                              arg_params=arg_params, aux_params=aux_params)

    except Exception, e:
        print(traceback.format_exc())
        predictor = None

    return predictor


def _load_cls(label_file):
    return tuple(e[-1] for e in net.load_labels(label_file))

# def net_init(config_str):


def net_init():
    # configs = json.loads(config_str)

    # tar_files_name = 'tar_files'
    # # load tar files
    # if tar_files_name not in configs:
    #     return dict(error='no field "tar_files"')

    # tar_files = configs[tar_files_name]
    # conf, err = net.parse_infer_config(tar_files)
    # if err:
    #     return dict(error=err)

    # params_file, sym_file, label_file = (conf.weight, conf.deploy_sym,
    #                                      conf.labels)

    # use_device_name = 'use_device'
    # if use_device_name not in configs:
    #     return dict(error='no field "use_device"')
    # use_device = configs[use_device_name]
    sym_file = "/tmp/eval/init/old-1222/deploy.symbol.json"
    params_file = "/tmp/eval/init/old-1222/weight.params"
    label_file = "/tmp/eval/init/old-1222/labels.csv"

    use_device = "GPU"
    threshold = 1e-3
    # thresholds = [0, 0.8, 0.8, 0.8, 0.7, 0.7, 1, 1, 1, 1, 1]
    thresholds = [0, 0.9, 0.8, 0.8,1,0.9, 0.9]
    # if 'custom_values' in configs:
    #     custom_values = configs['custom_values']
    #     if 'threshold' in custom_values:
    #         threshold = custom_values["threshold"]
    #     if 'thresholds' in custom_values:
    #         thresholds = custom_values['thresholds']

    ctx = [mx.gpu()] if use_device == 'GPU' else [
        mx.cpu()]  # TODO set the gpu/cpu
    classes = _make_synset(label_file)

    os.rename(sym_file, sym_file+'-symbol.json')
    os.rename(params_file, sym_file+'-0000.params')

    update_config("/tmp/eval/init/old-1222/resnet.yaml")

    return dict(
        error='',
        predictor=get_net(config, ctx, sym_file, config.TEST.test_epoch, True),
        classes=classes,
        threshold=threshold,
        thresholds=thresholds)


def generate_batch(im):
    """
    preprocess image, return batch
    :param im: cv2.imread returns [height, width, channel] in BGR
    :return:
    data_batch: MXNet input batch
    data_names: names in data_batch
    im_scale: float number
    """
    SHORT_SIDE = config.SCALES[0][0]
    LONG_SIDE = config.SCALES[0][1]
    PIXEL_MEANS = config.network.PIXEL_MEANS
    DATA_NAMES = ['data', 'im_info']

    im_array, im_scale = resize(im, SHORT_SIDE, LONG_SIDE)
    im_array = transform(im_array, PIXEL_MEANS)
    im_info = np.array(
        [[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
    data = [[mx.nd.array(im_array), mx.nd.array(im_info)]]
    data_shapes = [[('data', im_array.shape), ('im_info', im_info.shape)]]
    data_batch = mx.io.DataBatch(
        data=data, label=[None], provide_data=data_shapes, provide_label=[None])
    return data_batch, DATA_NAMES, [im_scale]


def _build_result(det, cls_name, cls_ind):
    ret = dict(index=cls_ind, score=float(det[-1]))
    ret['class'] = cls_name
    x1, y1, x2, y2 = det[:4]
    ret['pts'] = [
        [int(x1), int(y1)],
        [int(x2), int(y1)],
        [int(x2), int(y2)],
        [int(x1), int(y2)],
    ]

    return ret


def net_inference(model):
    """
    generate data_batch -> im_detect -> post process
    :param predictor: Predictor
    :param image_name: image name
    :return: None
    """

    # datas = json.loads(args)
    predictor = model['predictor']
    classes = model['classes']
    threshold = model['threshold']
    thresholds = model['thresholds']
    rets = []
    nms = py_nms_wrapper(config.TEST.NMS)
    box_voting = py_box_voting_wrapper(config.TEST.BOX_VOTING_IOU_THRESH, config.TEST.BOX_VOTING_SCORE_THRESH,
                                       with_nms=True)
    try:
        time_str = time.strftime(
            "%Y-%m-%d-%H-%M-%S", time.localtime())
        for i in sorted(os.listdir("/tmp/eval/init/images")):
            imageFile = os.path.join("/tmp/eval/init/images", i)
            print('-*-'*50)
            print(imageFile)
            _t1 = time.time()
            try:
                im = load_image(imageFile, 50.0)
            except ErrorBase as e:
                rets.append({"code": e.code, "message": e.message})
                continue

            data_batch, data_names, im_scale = generate_batch(im)
            scores, boxes, data_dict = im_detect(predictor,
                                                 data_batch,
                                                 data_names,
                                                 im_scale,
                                                 config)
            det_ret = []
            for cls_index, cls in enumerate(classes[1:], start=1):
                if len(cls) > 1:
                    cls_ind = int(cls[0])
                    cls_name = cls[1]
                else:
                    cls_ind = cls_index
                    cls_name = cls[0]
                cls_boxes = boxes[0][:, 4:8] if config.CLASS_AGNOSTIC else boxes[0][:,
                                                                                    4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[0][:, cls_ind, np.newaxis]
                if len(classes) <= len(thresholds):
                    threshold = thresholds[cls_ind]
                keep = np.where(cls_scores >= threshold)[0]
                dets = np.hstack((cls_boxes, cls_scores)).astype(
                    np.float32)[keep, :]
                # if "20170930_guns_1083.jpg" in imageFile:
                #     np.savetxt(fileOp, dets, delimiter=",")
                #     pass
                keep = nms(dets)
                # if "20170930_guns_1083.jpg" in imageFile:
                #     # print("end"*10)
                #     # print(dets[keep, :])
                #     # print('*'*100)
                #     # for i in dets:
                #     #     fileOp_1_op.write(i)
                #     #     fileOp_1_op.write('\n')
                #     np.savetxt(fileOp_1, dets[keep, :], delimiter=",")
                det_ret.extend(_build_result(det, cls_name, cls_ind)
                               for det in dets[keep, :])

            _t2 = time.time()
            print("inference image time : %f" % (_t2 - _t1))
            rets.append(
                dict(
                    code=0,
                    message=imageFile,
                    result=json.dumps(dict(detections=det_ret))))

    except Exception, e:
        print(traceback.format_exc())
    return infer_output_marshal(rets)


def net_inference(model):
    """
    generate data_batch -> im_detect -> post process
    :param predictor: Predictor
    :param image_name: image name
    :return: None
    """

    # datas = json.loads(args)
    predictor = model['predictor']
    classes = model['classes']
    threshold = model['threshold']
    thresholds = model['thresholds']
    rets = []
    nms = py_nms_wrapper(config.TEST.NMS)
    box_voting = py_box_voting_wrapper(config.TEST.BOX_VOTING_IOU_THRESH, config.TEST.BOX_VOTING_SCORE_THRESH,
                                       with_nms=True)
    try:
        time_str = time.strftime(
            "%Y-%m-%d-%H-%M-%S", time.localtime())
        fileOp = "/tmp/eval/init/20170930_guns_1083-begin-"+time_str+'.csv'
        fileOp_op = open(fileOp, 'w')
        time_str = time.strftime(
            "%Y-%m-%d-%H-%M-%S", time.localtime())
        fileOp_1 = "/tmp/eval/init/20170930_guns_1083-end-"+time_str+'.csv'
        fileOp_1_op = open(fileOp_1, 'w')
        fileOp_2 = "/tmp/eval/init/20170930_guns_1083-image-"+time_str+'.jpg'
        for i in sorted(os.listdir("/tmp/eval/init/images")):
            imageFile = os.path.join("/tmp/eval/init/images", i)
            try:
                im = load_image(imageFile, 50.0)
                # cv2.imwrite('fileOp_2', im)
                print(im[0, :])
                # np.savetxt(fileOp_2, im, delimiter=",")
            except ErrorBase as e:
                rets.append({"code": e.code, "message": e.message})
                continue

            data_batch, data_names, im_scale = generate_batch(im)
            print("*"*100)
            scores, boxes, data_dict = im_detect(predictor,
                                                 data_batch,
                                                 data_names,
                                                 im_scale,
                                                 config)
            det_ret = []
            for cls_index, cls in enumerate(classes[1:], start=1):
                if len(cls) > 1:
                    cls_ind = int(cls[0])
                    cls_name = cls[1]
                else:
                    cls_ind = cls_index
                    cls_name = cls[0]
                cls_boxes = boxes[0][:, 4:8] if config.CLASS_AGNOSTIC else boxes[0][:,
                                                                                    4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[0][:, cls_ind, np.newaxis]
                if len(classes) <= len(thresholds):
                    threshold = thresholds[cls_ind]
                keep = np.where(cls_scores >= threshold)[0]
                dets = np.hstack((cls_boxes, cls_scores)).astype(
                    np.float32)[keep, :]
                if "20170930_guns_1083.jpg" in imageFile:
                    # print(dets)
                    # print('*'*100)
                    # for i in dets:
                    #     fileOp_op.write(i)
                    #     fileOp_op.write('\n')
                    np.savetxt(fileOp, dets, delimiter=",")
                    pass
                keep = nms(dets)
                if "20170930_guns_1083.jpg" in imageFile:
                    # print("end"*10)
                    # print(dets[keep, :])
                    # print('*'*100)
                    # for i in dets:
                    #     fileOp_1_op.write(i)
                    #     fileOp_1_op.write('\n')
                    np.savetxt(fileOp_1, dets[keep, :], delimiter=",")
                det_ret.extend(_build_result(det, cls_name, cls_ind)
                               for det in dets[keep, :])

            rets.append(
                dict(
                    code=0,
                    message=imageFile,
                    result=json.dumps(dict(detections=det_ret))))

    except Exception, e:
        print(traceback.format_exc())
    return infer_output_marshal(rets)


def vis(imageName=None, bbox_list=None):
    im = cv2.imread(imageName, cv2.IMREAD_COLOR)
    color_black = (0, 0, 0)
    bbox_list = json.loads(bbox_list)
    for bbox_dict in bbox_list:
        print(bbox_dict)
        # bbox_dict = json.loads(bbox_dict)
        bbox = bbox_dict['pts']
        cls_name = bbox_dict['class']
        score = bbox_dict['score']
        cv2.rectangle(im, (bbox[0][0], bbox[0][1]), (bbox[2][0],
                                                     bbox[2][1]), color=color_black, thickness=3)
        cv2.putText(im, '%s %.3f' % (cls_name, score),
                    (bbox[0][0], bbox[0][1] + 15), color=color_black, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        out_file = os.path.join("/workspace/serving/python/evals/Anno",
                                imageName.split('/')[-1])
        cv2.imwrite(out_file, im)
    pass


def main():
    model = net_init()
    res = net_inference(model)
    classes = model['classes']
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    rfFile = "/tmp/eval/init/rf-result"+"-"+time_str+'.tsv'
    fileOp = open(rfFile, 'w')
    res = json.loads(res)
    res_list = res['results']
    for l in res_list:
        image_name = l['message'].split('/')[-1]
        bbox_list = json.dumps(json.loads(l['result'])['detections'])
        line = "%s\t%s\n" % (image_name, bbox_list)
        fileOp.write(line)
        if bbox_list == "[]":
            continue
        vis(imageName=l['message'], bbox_list=bbox_list)


if __name__ == '__main__':
    main()
