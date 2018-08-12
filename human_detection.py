# human_detection.py

import numpy as np
import tensorflow as tf
from yolo_model import YOLO
import cv2
from matplotlib import pyplot as plt
import time
import os, sys
import argparse


class HumanDetection():
    def __init__(self):
        print('Init')
        #select the checkpoint
        if False:
            self.path="utils/ckpt/tiny_yolo"
            self._type = 'TINY_VOC'
        else:
            self.path="utils/ckpt/yolov2"
            self._type = 'V2_VOC'

        
        with tf.Graph().as_default():
            self.img_in = tf.placeholder(tf.float32,[None, 416, 416, 3])
            self.clf = YOLO(self.img_in, yolotype= self._type)

            if not False:
                self.sess_type = tf.Session()
            else:
                self.sess_type = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
                                                            intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS'])))

            self.saver = tf.train.Saver()          
            self.saver.restore(self.sess_type, self.path)
        self.batch = 1


    def get_bbox(self, box_preds):
        # Tracking trigger
        done = False
        bb = []
        batch_addr = box_preds['batch_addr']
        boxes =box_preds['boxes']
        indices = box_preds['indices']
        class_names = box_preds['class_names']

        boxes = [boxes[i] for i in indices]
        class_names = [class_names[i] for i in indices]
        for i, b in enumerate(boxes):
            # idx  = batch_addr[i]
            print(str(class_names[i]))
            if class_names[i] == b'person':
                left = int(max(0, b[0]))
                top  = int(max(0, b[1]))
                right= int(min(415, b[2]))
                bot  = int(min(415, b[3]))

                # print((left, top, right, bot))
                bb.append(tuple([left, top, right-left, bot-top]))
        return bb


    def detect(self, img):
        print('detect')
        img3 = np.reshape(img, newshape=(1, 416, 416, 3))
        image = img3*0.003921569 # normalization
        box_preds = self.sess_type.run(self.clf.preds, {self.img_in: image})
        bbox = self.get_bbox(box_preds)

        # print(bbox)
        # if self.stop_detection:
        #     [x, y, w, h] = bbox
        #     cv2.imshow('hi', img4[y:y+h, x:x+w])
        #     ok = self.tracker.init(img4, bbox)
        return bbox

