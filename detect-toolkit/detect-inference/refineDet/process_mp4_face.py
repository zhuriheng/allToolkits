# -*- coding:utf-8 -*-
"""
    这个脚本用于处理视频
    读取视频，模型处理，输出结果
"""
import numpy as np
import sys
import os
import cv2
sys.path.insert(
    0, "/workspace/data/BK/refineDet-Dir/RefineDet/python")
import caffe
from argparse import ArgumentParser
import time
import json
import urllib
import random
import process_mp4_config


def parser_args():
    parser = ArgumentParser('process video script')
    parser.add_argument('--inputFileDir', dest='inputFileDir', help='local video in the folder',
                        default=None, type=str)
    return parser.parse_args()


class RefineDet_Face:
    def __init__(self, config):
        self.net = config['net']
        self.image_size = config['image_size']
        self.batch_size = config['batch_size']
        self.label_list = config['label_list']
        self.neel_label_dict = config['neel_label_dict']
        self.need_label_thresholds = config['need_label_thresholds']

    def preProcess(self, oriImage):
        img = cv2.resize(
            oriImage, (self.image_size, self.image_size))
        img = img.astype(np.float32, copy=False)
        img = img - np.array([[[104, 117, 123]]])
        img = img * 0.017  # 根据训练网络结构中  transform_param 是否 使用 scala
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        return img

    def inference(self, oriImage_list, time_list):
        if len(oriImage_list) > self.batch_size:
            print("the input image length > batch_size : %d -- %d" %
                  (len(oriImage_list), self.batch_size))
        batch_image_h_w = []
        batch_image_data = []
        for oriImg in oriImage_list:
            img = self.preProcess(oriImage=oriImg)
            batch_image_data.append(img)
            h_w = [oriImg.shape[0], oriImg.shape[1]]
            batch_image_h_w.append(h_w)
        for index, i_data in enumerate(batch_image_data):
            self.net.blobs['data'].data[index] = i_data
        _t1 = time.time()
        output = self.net.forward()
        _t2 = time.time()
        print("inference time : %f" % (_t2-_t1))
        res = self.post_eval(time_list, batch_image_h_w, output)
        # res is list
        write_list, vis_list = self.process_res(oriImage_list, res_list=res)
        return [write_list, vis_list]

    def post_eval(self, time_list, batch_image_h_w, output):
        cur_batchsize = len(batch_image_h_w)
        print("cur_batchsize is : %d" % (cur_batchsize))
        _t1 = time.time()
        # output_bbox_list : bbox_count * 7
        output_bbox_list = output['detection_out'][0][0]
        image_result_dict = dict()  # image_id : bbox_list
        for i_bbox in output_bbox_list:
            image_id = int(i_bbox[0])
            if image_id >= cur_batchsize:
                break
            # image_data = net.images[image_id]
            w = batch_image_h_w[image_id][1]
            h = batch_image_h_w[image_id][0]
            class_index = int(i_bbox[1])
            if class_index not in self.neel_label_dict.keys():
                continue
            #if class_index < 1 or class_index >= 6:
            #    continue
            score = float(i_bbox[2])
            #if score < THRESHOLDS[class_index]:
            #    continue
            if score < self.need_label_thresholds.get(class_index):
                continue
            bbox_dict = dict()
            bbox_dict['index'] = class_index
            bbox_dict['class'] = self.label_list[class_index]
            bbox_dict['score'] = score
            bbox = i_bbox[3:7] * np.array([w, h, w, h])
            bbox_dict['pts'] = []
            xmin = int(bbox[0]) if int(bbox[0]) > 0 else 0
            ymin = int(bbox[1]) if int(bbox[1]) > 0 else 0
            xmax = int(bbox[2]) if int(bbox[2]) < w else w
            ymax = int(bbox[3]) if int(bbox[3]) < h else h
            bbox_dict['pts'].append([xmin, ymin])
            bbox_dict['pts'].append([xmax, ymin])
            bbox_dict['pts'].append([xmax, ymax])
            bbox_dict['pts'].append([xmin, ymax])
            if image_id in image_result_dict.keys():
                the_image_bbox_list = image_result_dict.get(image_id)
                the_image_bbox_list.append(bbox_dict)
                pass
            else:
                the_image_bbox_list = []
                the_image_bbox_list.append(bbox_dict)
                image_result_dict[image_id] = the_image_bbox_list
        resps = []
        for image_id in range(cur_batchsize):
            if image_id in image_result_dict.keys():
                res_list = image_result_dict.get(image_id)
            else:
                res_list = []
            result = {"detections": res_list}
            resps.append(
                {"code": 0, "message": time_list[image_id], "result": result})
        _t2 = time.time()
        print("post_eval the batch time is  : %f" % (_t2 - _t1))
        return resps

    def process_res(self, oriImage_list, res_list=None):
        write_list = []
        vis_list = []
        for index, image_res in enumerate(res_list):
            origim = oriImage_list[index]
            # image_res is dict
            bbox_list = image_res['result']['detections']
            vis_res = self.vis(
                origim, timeFlag=image_res['message'], bbox_list=bbox_list)
            vis_list.append(vis_res)
            write_res = self.write_rf(image_res=image_res)
            write_list.append(write_res)
        return [write_list, vis_list]
        pass

    def write_rf(self, image_res=None):
        bbox_list = json.dumps(image_res['result']['detections'])
        image_name = image_res['message']
        line = "%s\t%s\n" % (image_name, bbox_list)
        return line

    def vis(self, origim, timeFlag, bbox_list):
        im = origim
        color_black = (0, 0, 0)
        color = (random.randint(0, 256), random.randint(
            0, 256), random.randint(0, 256))
        for bbox_dict in bbox_list:
            bbox = bbox_dict['pts']
            cls_name = bbox_dict['class']
            score = bbox_dict['score']
            cv2.rectangle(im, (bbox[0][0], bbox[0][1]), (bbox[2][0],
                                                         bbox[2][1]), color=color, thickness=3)
            cv2.putText(im, '%s %.3f' % (cls_name, score),
                        (bbox[0][0], bbox[0][1] + 15), color=color_black, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        return im
    pass


def processVideo(refinede_class, videoPath=None):
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    output_video_path = None
    if '.' in os.path.basename(videoPath):
        output_video_path = videoPath[:videoPath.rfind('.')]+'-detect-face.mp4'
    else:
        output_video_path = videoPath+'-detect-face.mp4'
    vwrite = cv2.VideoWriter(
        output_video_path, fourcc, fps, size)
    success, img = cap.read()
    count = 1
    while (success):
        time_list = []
        time_list.append(count)
        oriImage_list = []
        oriImage_list.append(img)
        write_list, vis_list = refinede_class.inference(
            oriImage_list, time_list)
        for write_list_i in write_list:
            print(write_list_i)
        for vis_result_i in vis_list:
            vwrite.write(vis_result_i)
        success, img = cap.read()
        count += 1
    cap.release()
    vwrite.release()
    pass


def getAllVideoFilePath(basePath=None):
    pass


def initModel_refinedet():
    config = process_mp4_config.face_detect_config
    caffe.set_mode_gpu()
    caffe.set_device(config['gpuId'])
    deployName = config['modelParam']['deploy']
    modelName = config['modelParam']['weight']
    labelName = config['modelParam']['label']
    image_size = config['image_size']
    bathc_size = config['modelParam']['batch_size']
    neel_label_dict = config['need_label_dict']
    need_label_thresholds = config['need_label_thresholds']
    net = caffe.Net(deployName, modelName, caffe.TEST)
    label_list = None
    with open(labelName, 'r') as f:
        label_list = [i.strip().split(',')[-1]
                      for i in f.readlines() if i.strip()]
    res_dict = {
        'net': net,
        'batch_size': bathc_size,
        'image_size': image_size,
        'label_list': label_list,
        'neel_label_dict': neel_label_dict,
        'need_label_thresholds': need_label_thresholds
    }
    return res_dict


def main():
    refinedet_dict = initModel_refinedet()
    refinede_class = RefineDet_Face(refinedet_dict)
    processVideo(
        refinede_class, videoPath="/workspace/data/BK/process-mp4/data/car2.mp4")
    pass


if __name__ == '__main__':
    main()
