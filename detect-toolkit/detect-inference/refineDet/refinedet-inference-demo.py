# -*- coding:utf -*-
import numpy as np
import sys
import os
import cv2
sys.path.insert(
    0, "/workspace/data/BK/terror-det-refineDet-Dir/refineDet_Dir/RefineDet/python")
import caffe
from argparse import ArgumentParser
import time
import json
import urllib
from google.protobuf import text_format
from caffe.proto import caffe_pb2
"""
    bk terror caffe detect model , refineDet res 18 model
    这个脚本是用来推理 refinedet 模型。
    ONE_BATCH_SIZE = 16 （按照 在deploy文件中设置的 batch_size 来改变这个值）
    thresholds  这个是用来控制 各个类别的。根据需求动态修改
    保存结果为 检测的 回归测试 格式
    weiboimg_wordlib_0-20170927_45944.jpg   [{"index": 2, "score": 0.997883141040802, "pts": [[9, 14], [568, 14], [568, 149], [9, 149]], "class": "guns"}, {"index": 2, "score": 0.9955806136131287, "pts": [[22, 112], [556, 112], [556, 302], [22, 302]], "class": "guns"}]

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
    parser.add_argument('--visFlag', dest='visFlag', help='visulize the detect bbox',
                        default=0, type=int) # 0 : not visulize , 1 : visulize
    # parser.add_argument('--threshold', dest='threshold',
    #                     default=0.1, type=float)
    return parser.parse_args()

RESULT_FILE_OP = None  # 用于保存结果文件
VISSAVEDIR = None # 该目录用于存放 检测结果图片（原图+bbox)
def temp_init():
    global RESULT_FILE_OP
    global THRESHOLDS  # 各个类别的阈值
    global URLFLAG   # 图片是否是url
    global IMAGE_SIZE # 处理图片的大小
    global ONE_BATCH_SIZE  # batch size 的大小 （ 要和 deploy 的设置一致）
    global VISSAVEDIR 
    ONE_BATCH_SIZE = 16
    URLFLAG = True if args.urlfileName else False
    IMAGE_SIZE = 320
    THRESHOLDS = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]
    if args.urlfileName:
        result_file = args.urlfileName+'-' + \
            str(args.urlfileName_beginIndex)+"-result"
    elif args.localImageNameFile:
        result_file = args.localImageNameFile+"-result"
    if args.visFlag == 1:
        VISSAVEDIR = result_file + "-vis"
        if not os.path.exist(VISSAVEDIR):
            os.makedirs(VISSAVEDIR)
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    result_file = result_file+"-"+time_str+'.tsv'
    RESULT_FILE_OP = open(result_file, 'w')
    pass

"""
0 : "background"
1 : "guns"
2 : "knives"
3 : "tibetan flag"
4 : "islamic flag"
5 : "isis flag"
6 : "not terror"
"""


def post_eval(input_image, batch_image_path, batch_image_h_w,output, reqs=None):
    global THRESHOLDS
    global CLS_LABEL_LIST
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
    # cur_batchsize = len(output['detection_out']) # len(output['detection_out'])  always 1
    cur_batchsize = len(input_image)
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
        if class_index < 1 or class_index >= 6:
            continue
        score = float(i_bbox[2])
        if score < THRESHOLDS[class_index]:
            continue
        bbox_dict = dict()
        bbox_dict['index'] = class_index
        bbox_dict['class'] = CLS_LABEL_LIST[class_index]
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
            {"code": 0, "message": batch_image_path[image_id], "result": result})
    _t2 = time.time()
    print("post_eval the batch time is  : %f\n", _t2 - _t1)
    return resps


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
    global CLS_LABEL_LIST  # 类别 label
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
    global ONE_BATCH_SIZE
    batch_image_data = []
    batch_image_path = []
    batch_image_h_w = []
    batch_count = 0
    for image_path in imageList:
        oriImg = readImage_fun(isUrlFlag=urlFlag, imagePath=image_path)
        if np.shape(oriImg) == ():
            print("ReadImageError : %s" % (image_path))
            continue
        if oriImg.shape[2] != 3:
            print("%s channel is not 3" % (image_path))
	    img = preProcess(oriImage=oriImg)
        batch_image_data.append(img)
        batch_image_path.append(image_path)
        h_w = [oriImg.shape[0], oriImg.shape[1]]
        batch_image_h_w.append(h_w)
        batch_count += 1
        if batch_count == ONE_BATCH_SIZE:
            print("batch_count : %d" % (batch_count))
            _t1 = time.time()
            for index, i_data in enumerate(batch_image_data):
                net_cls.blobs['data'].data[index] = i_data
            output = net_cls.forward()
            _t2 = time.time()
            print("inference time : %f" % (_t2-_t1))
            # postProcess_batch(output_batch=detection_out_batch_result,
            #                  imagePath_batch=batch_image_path, img_shape_batch=batch_image_h_w)
            res = post_eval(batch_image_data, batch_image_path, batch_image_h_w,
                            output, reqs=None)
            # res is list
            process_res(res_list=res)
            batch_image_data = []
            batch_image_path = []
            batch_image_h_w = []
            batch_count = 0

    # process not enough 16
    last_batch_size = len(batch_image_data)
    if last_batch_size == 0:
        return
    else:
        print("batch_count : %d" % (last_batch_size))
        a = time.time()
        # in_shape = net_cls.blobs['data'].data.shape
        # in_shape = (last_batch_size, in_shape[1], in_shape[2], in_shape[3])
        _t1 = time.time()
        # net_cls.blobs['data'].reshape(*in_shape)
        # net_cls.reshape()
        _t2 = time.time()
        print("net reshape time : %f" % (_t2-_t1))
        for index, i_data in enumerate(batch_image_data):
            net_cls.blobs['data'].data[index] = i_data
        output = net_cls.forward()
        b = time.time()
        print("not 16 image  time : %f" % (b-a))
        # postProcess_batch(output_batch=detection_out_batch_result,
        #                   imagePath_batch=batch_image_path, img_shape_batch=batch_image_h_w)
        res = post_eval(batch_image_data, batch_image_path,
                        batch_image_h_w, output, reqs=None)
        process_res(res_list=res)
    return
def process_res(res_list=None):
    for image_res in res_list:
        # image_res is dict
        bbox_list = image_res['result']['detections']
        if args.visFlag == 1:
            vis(imageName=image_res['message'], bbox_list=bbox_list)
        write_rf(image_res=image_res)
    pass


def write_rf(image_res=None):
    global rg_file_op
    bbox_list = json.dumps(image_res['result']['detections'])
    image_name = image_res['message'].split('/')[-1]
    line = "%s\t%s\n" % (image_name, bbox_list)
    rg_file_op.write(line)
    rg_file_op.flush()
    pass


def vis(imageName=None, bbox_list=None):
    global VISSAVEDIR 
    im = cv2.imread(imageName, cv2.IMREAD_COLOR)
    out_file = os.path.join(VISSAVEDIR,
                            imageName.split('/')[-1])
    color_black = (0, 0, 0)
    for bbox_dict in bbox_list:
        # bbox_dict = json.loads(bbox_dict)
        bbox = bbox_dict['pts']
        cls_name = bbox_dict['index']
        score = bbox_dict['score']
        cv2.rectangle(im, (bbox[0][0], bbox[0][1]), (bbox[2][0],
                                                     bbox[2][1]), color=color_black, thickness=3)
        cv2.putText(im, '%s %.3f' % (cls_name, score),
                    (bbox[0][0], bbox[0][1] + 15), color=color_black, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    cv2.imwrite(out_file, im)
    pass


args = parser_args()
CLS_LABEL_LIST = None
THRESHOLDS = None
URLFLAG = None
IMAGE_SIZE = None
ONE_BATCH_SIZE = None


def main():
    temp_init()
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
