#coding=utf-8
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

Result_file = None
CLS_LABEL_LIST = None
THRESHOLD = None
URLFLAG = None
IMAGE_SIZE = None
ONE_BATCH_SIZE = None
VISPATH = None

def init_configs():
    global people_result_file_OP
    global rg_file_op
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
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    rg_file = people_result_file+"-"+time_str
    rg_file_op = open(rg_file, 'w')
    pass


def postProcess_batch(output_batch=None, imagePath_batch=None, img_shape_batch=None):
    image_result_bath_dict = dict()
    output_bbox = output_batch[0][0]
    for i_bbox in output_bbox:
        image_index = i_bbox[0]
        image_index = int(image_index)
        if image_index >= len(imagePath_batch):
            continue
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


"""
item {
  name: "none_of_the_above"
  label: 0
  display_name: "background"
}
item {
  name: "guns"
  label: 1
  display_name: "guns"
}
item {
  name: "knives"
  label: 2
  display_name: "knives"
}
item {
  name: "tibetan flag"
  label: 3
  display_name: "tibetan flag"
}
item {
  name: "islamic flag"
  label: 4
  display_name: "islamic flag"
}
item {
  name: "isis flag"
  label: 5
  display_name: "isis flag"
}
item {
  name: "not terror"
  label: 6
  display_name: "not terror"
}
"""

# def post_eval(net, output, thresholds, reqs=None):


def post_eval(input_image, batch_image_path, batch_image_h_w, output, thresholds, reqs=None):
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
        if score < thresholds[class_index]:
            continue
        bbox_dict = dict()
        bbox_dict['index'] = class_index
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
    print("post: %f\n", _t2 - _t1)
    # CTX.logger.info("post: %f\n", _t2 - _t1)
    # monitor_rt_post().observe(_t2 - _t1)
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
    global CLS_LABEL_LIST
    global THRESHOLD
    global URLFLAG
    global IMAGE_SIZE
    global ONE_BATCH_SIZE
    ONE_BATCH_SIZE = 16
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
    thresholds = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0]
    batch_image_data = []
    batch_image_path = []
    batch_image_h_w = []
    batch_count = 0
    batch_size = 16
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
        if batch_count == batch_size:
            print("batch_count : %d" % (batch_count))
            _t1 = time.time()
            for index, i_data in enumerate(batch_image_data):
                net_cls.blobs['data'].data[index] = i_data
            output = net_cls.forward()
            detection_out_batch_result = output['detection_out']
            _t2 = time.time()
            print("16 image time : %f" % (_t2-_t1))
            postProcess_batch(output_batch=detection_out_batch_result,
                              imagePath_batch=batch_image_path, img_shape_batch=batch_image_h_w)
            res = post_eval(batch_image_data, batch_image_path, batch_image_h_w,
                            output, thresholds, reqs=None)
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
                        batch_image_h_w, output, thresholds, reqs=None)
        print(res)
        process_res(res_list=res)
    return


time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
rfFile = "/tmp/eval/init/rf-result"+"-"+time_str+'.tsv'


def process_res(res_list=None):
    for image_res in res_list:
        # image_res is dict
        # res = json.loads(res)
        # image_res = json.loads(image_res)
        bbox_list = image_res['result']['detections']
        if VISPATH :
            vis(imageName=image_res['message'],
                bbox_list=bbox_list, savePath=VISPATH)
        write_rf(image_res=image_res)
    pass


def write_rf(image_res=None):
    global rg_file_op
    bbox_list = json.dumps(image_res['result']['detections'])
    image_name = image_res['message'].split('/')[-1]
    line = "%s\t%s\n" % (image_name, bbox_list)
    rg_file_op.write(line)
    pass


def vis(imageName=None, bbox_list=None, savePath=VISPATH):
    im = cv2.imread(imageName, cv2.IMREAD_COLOR)
    out_file = os.path.join(savePath,
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

def main():
    init_configs()
    net_cls = init_models()
    imageList = getImagePath()
    infereneAllImage(net_cls=net_cls, imageList=imageList, urlFlag=URLFLAG)


if __name__ == '__main__':
    main()


"""
python refinedet-inference.py \
--urlfileName ./data/test_url.list \
--modelBasePath ./models/ \
--modelName terror-res18-320x320-t2_iter_130000.caffemodel \
--deployFileName deploy.prototxt \
--labelFileName  labelmap_bk.prototxt  \
--gpu 0

"""
