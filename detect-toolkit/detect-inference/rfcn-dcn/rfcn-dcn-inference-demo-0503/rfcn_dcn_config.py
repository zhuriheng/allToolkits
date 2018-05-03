rfcn_dcn_config = {
    "modelParam": {
        "weight": "/workspace/data/BK/process-mp4/models/ResNet/coco/refinedet_resnet101_512x512/coco_refinedet_resnet101_512x512_final.caffemodel",
        "deploy": "/workspace/data/BK/process-mp4/models/ResNet/coco/refinedet_resnet101_512x512/deploy.prototxt",
        "label": "/workspace/data/BK/process-mp4/models/ResNet/coco/refinedet_resnet101_512x512/coco_label.csv",
        "batch_size": 1
    },
    "gpuId": 0,
    'image_size': 512,
    'need_label_dict': {
        1: 'person',
        3: 'car',
        4: 'motorbike',
        6: 'bus',
        8: 'truck'
    },
    'need_label_thresholds': {
        1: 0.4,
        3: 0.4,
        4: 0.4,
        6: 0.4,
        8: 0.4
    }
}
