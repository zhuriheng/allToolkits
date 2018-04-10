# -*- coding:utf-8 -*-
import os
import sys
import json


def get_jsonList_line_labelInfo(line=None):
    """
    value is list : 
        the element in the list is : 
            bbox list : 
                [
                    {"class":"knives_true","bbox":[[43,105],[138,105],[138,269],[43,269]],"ground_truth":true},
                    {"class":"guns_true","bbox":[[62,33],[282,33],[282,450],[62,450]],"ground_truth":true},
                    {"class":"guns_true","bbox":[[210,5],[399,5],[399,487],[210,487]],"ground_truth":true}
                ]
    return key:value 
            key is url
            value : label info ()
    """
    key = None
    value = None # value is  all bbox info list , element is dict
    line_dict = json.loads(line)
    key = line_dict['url']
    if line_dict['label'] == None or len(line_dict['label']) == 0:
        return key, None
    label_dict = line_dict['label'][0]
    if label_dict['data'] == None or len(label_dict['data']) == 0:
        return key, None
    data_dict_list = label_dict['data']
    label_bbox_list_elementDict = []
    for bbox in data_dict_list:
        if 'class' not in bbox or bbox['class'] == None or len(bbox['class']) == 0:
            continue
        label_bbox_list_elementDict.append(bbox)
    if len(label_bbox_list_elementDict) == 0:
        value = None
    else:
        value = label_bbox_list_elementDict
    return key, value


def get_IOU(bbox_a=None, bbox_b=None):
    """
    自定义函数，计算两矩形 IOU，传入为 bbox 是 [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
    """
    Reframe = [bbox_a[0][0], bbox_a[0][1], bbox_a[2][0], bbox_a[2][1]]
    GTframe = [bbox_b[0][0], bbox_b[0][1], bbox_b[2][0], bbox_b[2][1]]
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1, x2+width2)
    startx = min(x1, x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1, y2+height2)
    starty = min(y1, y2)
    height = height1+height2-(endy-starty)

    if width <= 0 or height <= 0:
        ratio = 0  # 重叠率为 0
    else:
        Area = width*height  # 两矩形相交面积
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    # return IOU
    return ratio


def mergeAllJsonListFileToOneFile(inputDir=None, tempSaveDir=None):
    """
        这个函数的作用：将要处理的多个 jsonlist 文件合并成一个。
        合并过程中：如果 jsonlist 的 url  相同，则去重
        去重规则是：保留标注 bbox 多的
    """
    anno_json_info_dict = {}  # image_name:annoInfo
    for json_file in sorted(os.listdir(inputDir)):
        if not json_file.endswith('.json') or json_file[0] == '.' or "all_json_file" in json_file:
            print("%s is not a json file , so not process" % (json_file))
            continue
        json_file = os.path.join(inputDir, json_file)
        with open(json_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) <= 0:
                    continue
                url, _ = get_jsonList_line_labelInfo(
                    line=line)
                if url in anno_json_info_dict:
                    oldLine = anno_json_info_dict.get(url)
                    newLine = line
                    _, new_value = get_jsonList_line_labelInfo(
                        line=newLine)
                    _, old_value = get_jsonList_line_labelInfo(
                        line=oldLine)
                    if len(new_value) > len(old_value):
                        anno_json_info_dict[url] = newLine
                    pass
                else:
                    # line contains '\n' in the end
                    anno_json_info_dict[url] = line
    all_json_file_name = os.path.join(tempSaveDir, 'all_json_file.json')
    with open(all_json_file_name, 'w') as f:
        for key in anno_json_info_dict:
            f.write(anno_json_info_dict[key])
            f.write('\n')
    return all_json_file_name
