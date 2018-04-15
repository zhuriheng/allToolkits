class Config:
    PLATFORM = "GPU"
    GPU_ID = 0
    DEPLOY_FILE = '../models/deploy.prototxt'
    MODEL_FILE = '../models/weight.caffemodel'
    LABEL_FILE = '../models/labels.csv'
    IMAGE_SIZE = 320
    THRESHOLD= 0.1
