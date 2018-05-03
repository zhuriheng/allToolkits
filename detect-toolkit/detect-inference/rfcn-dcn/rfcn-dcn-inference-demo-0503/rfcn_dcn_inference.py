# -*- coding: utf-8 -*-
"""
rfcn-dcn-demo script
change log :
    18-03-12:
        add configYamlFile param
"""
import _init_paths
import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
import random
import urllib
import json
import copy
import rfcn_dcn_config

def parse_args():
    parser = argparse.ArgumentParser(description='rfcn-dcn-inference demo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--localImageListFile', help='local images  list file', default=None, type=str)
    parser.add_argument(
        '--urlImageListFile', help='input images url list file', default=None, type=str)
    parser.add_argument('--gpuId', required=True,dest='gpuId', help='the id of gpu', type=str)
    # outputFileFlag : 1 是回归测试文件，2 是 labex 格式的输出文件
    parser.add_argument('--outputFileFlag', required=True,
                        help='out put file flag', type=int)
    # 0 no visualize  , 1 visualize
    parser.add_argument(
        '--visualizeFlag', help='visualize the detect reuslt', type=int, default=False)
    parser.add_argument('--beginLineNum',
                        help='beginLineNum', type=int, default=0)
    args = parser.parse_args()
    return args


def readImage_fun(isUrlFlag=False, imagePath=None):
    im = None
    if isUrlFlag:
        try:
            data = urllib.urlopen(imagePath.strip()).read()
            nparr = np.fromstring(data, np.uint8)
            if nparr.shape[0] < 1:
                im = None
        except:
            print("Read Url Exception : %s" % (imagePath))
            im = None
        else:
            im = cv2.imdecode(nparr, 1)
        finally:
            return im
    else:
        im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if np.shape(im) == ():
        print("waringing info : %s can't be read" % (imagePath))
        return None
    return im


def show_boxes_write_rg(fileOp=None, image_name=None, im=None, dets=None, classes=None, vis=False, scale=1.0):
    color_white = (255, 255, 255)
    color_black = (0, 0, 0)
    # write to terror det rg tsv file
    thresholds = [0, 0.8, 0.8, 0.8, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0]
    imageName = image_name
    writeInfo = []
    for cls_idx, cls_name in enumerate(classes[1:], start=1):
        if cls_idx > 5:
            continue
        write_bbox_info = {}
        write_bbox_info['class'] = cls_name
        write_bbox_info['index'] = cls_idx

        cls_dets = dets[cls_idx-1]
        color = (random.randint(0, 256), random.randint(
            0, 256), random.randint(0, 256))
        for det in cls_dets:
            bbox = det[:4] * scale
            score = det[-1]
            if float(score) < thresholds[cls_idx]:
                continue
            bbox = map(int, bbox)
            one_bbox_write = copy.deepcopy(write_bbox_info)
            bbox_position_list = []
            bbox_position_list.append([bbox[0], bbox[1]])
            bbox_position_list.append([bbox[2], bbox[1]])
            bbox_position_list.append([bbox[2], bbox[3]])
            bbox_position_list.append([bbox[0], bbox[3]])
            one_bbox_write["pts"] = bbox_position_list
            # one_bbox_write["score"] = float(score)
            one_bbox_write["score"] = float(score)
            writeInfo.append(one_bbox_write)
            if vis:
                cv2.rectangle(
                    im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color_black, thickness=3)
                cv2.putText(im, '%s %.3f' % (cls_name, score), (
                    bbox[0], bbox[1] + 15), color=color_black, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    fileOp.write("%s\t%s" % (imageName.split('/')[-1], json.dumps(writeInfo)))
    fileOp.write('\n')
    fileOp.flush()
    if vis:
        out_file = os.path.join(args.visualizeOutPutPath,
                                imageName.split('/')[-1])
        cv2.imwrite(out_file, im)
    pass


def show_boxes_write_labelx(fileOp=None, image_name=None, im=None, dets=None, classes=None, vis=False, scale=1.0):
    color_white = (255, 255, 255)
    color_black = (0, 0, 0)
    # write to terror det rg tsv file
    imageName = image_name
    writeInfo = dict()
    writeInfo['url'] = image_name
    writeInfo['type'] = "image"
    label_dict = dict()
    general_d_dict = dict()
    bbox_list = []
    detect_dict = dict()
    color = (random.randint(0, 256), random.randint(
        0, 256), random.randint(0, 256))
    for cls_idx, cls_name in enumerate(classes[1:], start=1):
        if cls_name == "not terror":
            continue
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            score = det[-1]
            bbox = map(int, bbox)
            one_bbox = dict()
            one_bbox['class'] = cls_name
            pts_list = []
            pts_list.append([bbox[0], bbox[1]])
            pts_list.append([bbox[2], bbox[1]])
            pts_list.append([bbox[2], bbox[3]])
            pts_list.append([bbox[0], bbox[3]])
            one_bbox["pts"] = pts_list
            bbox_list.append(one_bbox)
            if vis:
                cv2.rectangle(
                    im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=3)
                cv2.putText(im, '%s %.3f' % (cls_name, score), (
                    bbox[0], bbox[1] + 15), color=color_black, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    if len(bbox_list) > 0:
        general_d_dict['bbox'] = bbox_list
        detect_dict['general_d'] = general_d_dict
        label_dict['detect'] = detect_dict
        writeInfo['label'] = label_dict
        fileOp.write("%s" % (json.dumps(writeInfo)))
        fileOp.write('\n')
        fileOp.flush()
    if vis:
        out_file = os.path.join(args.visualizeOutPutPath,
                                imageName.split('/')[-1])
        out_file = out_file[:out_file.rfind('.')]+'.jpg'
        cv2.imwrite(out_file, im)
    pass


def show_boxes(isUrlFlag=None, im_name=None, dets=None, classes=None, scale=1, vis=False, fileOp=None, flag=1):
    im = None
    if vis:
        im = readImage_fun(isUrlFlag=isUrlFlag, imagePath=im_name)
    if flag == 1:
        show_boxes_write_rg(fileOp=fileOp, image_name=im_name,
                            im=im, dets=dets, classes=classes, vis=vis)
    elif flag == 2:
        show_boxes_write_labelx(fileOp=fileOp, image_name=im_name,
                                im=im, dets=dets, classes=classes, vis=vis)
    pass


def process_one_batch_images_fun(isUrlFlag=False, one_batch_images_list=None, init_model_param=None, fileOp=None, vis=False):
    # init_model_param list : [sym, arg_params, aux_params]

    num_classes = 11  # 0 is background,
    # classes = ['tibetan flag', 'guns', 'knives',
    #            'not terror', 'islamic flag', 'isis flag']
    classes = ['__background__',
               'islamic flag', 'isis flag', 'tibetan flag', 'knives_true', 'guns_true',
               'knives_false', 'knives_kitchen',
               'guns_anime', 'guns_tools',
               'not terror']
    image_names = one_batch_images_list
    if len(image_names) <= 0:
        return
    all_can_read_image = []
    data = []
    for im_name in image_names:
        #print("process : %s"%(im_name))
        im = readImage_fun(isUrlFlag=isUrlFlag, imagePath=im_name)
        # 判断 这个图片是否可读
        if np.shape(im) == ():
            print("ReadImageError : %s" % (im_name))
            continue
        all_can_read_image.append(im_name)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size,
                              stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array(
            [[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names]
            for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max(
        [v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])]
                    for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]

    predictor = Predictor(init_model_param[0], data_names, label_names,
                          context=[mx.gpu(int(args.gpuId))], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=init_model_param[1], aux_params=init_model_param[2])
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

    for idx, im_name in enumerate(all_can_read_image):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx, provide_data=[
                                     [(k, v.shape) for k, v in zip(data_names, data[idx])]], provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2]
                  for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, data_dict = im_detect(
            predictor, data_batch, data_names, scales, config)
        boxes = boxes[0].astype('f')
        scores = scores[0].astype('f')
        dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:,
                              4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > args.threshold, :]
            dets_nms.append(cls_dets)
        print('testing {} {:.4f}s'.format(im_name, toc()))
        show_boxes(isUrlFlag=isUrlFlag, im_name=im_name, dets=dets_nms,
                   classes=classes, scale=1, vis=vis, fileOp=fileOp, flag=args.outputFileFlag)
    print('process one batch images done')
    pass


def init_detect_model():
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_rfcn_dcn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)
    arg_params, aux_params = load_param(
        cur_path + '/demo/models/' + ('rfcn_voc'), int(args.epoch), process=True)
    return [sym, arg_params, aux_params]


def process_image_fun(isUrlFlag=False, imagesPath=None, fileOp=None, vis=False):
    # init rfcn dcn detect model (mxnet)
    model_params_list = init_detect_model()
    beginCount = args.beginProcessLineNum
    endCount = beginCount
    for i in range(len(imagesPath)/ONE_BATCH_IMAGE_COUNT):
        endCount += ONE_BATCH_IMAGE_COUNT
        tempFileList = imagesPath[beginCount:endCount]
        process_one_batch_images_fun(
            isUrlFlag=isUrlFlag, one_batch_images_list=tempFileList, init_model_param=model_params_list, fileOp=fileOp, vis=vis)
        print("line %d process done" % (endCount))
        beginCount = endCount

    tempFileList = imagesPath[beginCount:]
    process_one_batch_images_fun(
        isUrlFlag=isUrlFlag, one_batch_images_list=tempFileList, init_model_param=model_params_list, fileOp=fileOp, vis=vis)
    print("the file  process done")
    pass


def main():
    update_config(os.path.join(cur_path, 'demo/models', args.configYamlFile))
    if args.visualize:
        if not args.visualizeOutPutPath:
            args.visualizeOutPutPath = os.path.join(
                cur_path, 'visualizeOutPutPath')
        if not os.path.exists(args.visualizeOutPutPath):
            os.makedirs(args.visualizeOutPutPath)
    if not os.path.exists(os.path.split(args.outputFilePath)[0]):
        os.makedirs(os.path.split(args.outputFilePath)[0])
    fileOp = open(args.outputFilePath+"-"+str(args.epoch) +
                  "-threshold-"+str(args.threshold), 'a+')  # 追加的方式，如果不存在就创建
    need_process_images_path_list = []
    if args.localImageListFile:
        basePath = ''
        if args.localImageBasePath:
            basePath = args.localImageBasePath
        with open(args.localImageListFile, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.endswith('.jpg'):
                    pass
                else:
                    line = line+'.jpg'
                image_path = os.path.join(basePath, line)
                need_process_images_path_list.append(image_path)
        process_image_fun(
            isUrlFlag=False, imagesPath=need_process_images_path_list, fileOp=fileOp, vis=args.visualize)
    elif args.urlImageListFile:
        with open(args.urlImageListFile, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                need_process_images_path_list.append(line)
        process_image_fun(
            isUrlFlag=True, imagesPath=need_process_images_path_list, fileOp=fileOp, vis=args.visualize)
    pass


args = parse_args()
if __name__ == '__main__':
    print(args)
    main()


"""

"""
