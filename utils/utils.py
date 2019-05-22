import os
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
import numpy as  np
import sys
from random import random
from tqdm import tqdm


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def normalize(images):
    return (images.astype(np.float32)/255.0)

def denormalize(images):
    images = np.clip(images,a_min= 0.0001, a_max = 0.999)
    return (images*255.0).astype(np.uint8)
