#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import cv2

import torch

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

import argparse
import os
import time

import mmglobal

from timeit import time
import numpy as np

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import generate_detections as gdet
import CheckCrossLine
import newLine
import imutils.video

from collections import Counter
from collections import deque 
import datetime
import math

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo2!")
    parser.add_argument(
        "demo2", default="image", help="demo2 type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo2 camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        a = [bboxes, scores, cls]
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, a


def image_demo2(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image, bboxes, scores, cls, cls_conf = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo2(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo2 == "video" else args.camid) #video
    #cap = cv2.VideoCapture("https://camerai1.iticfoundation.org/hls/pty02.m3u8") #url real-time
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo2 == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mkv")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (int(width), int(height))
    )
    
    # Hui: Create Window
    #win_name = 'Video detection'
    #cv2.namedWindow(win_name)

    mmglobal.frame_count = 0;
    
    # Definition of the parameters
    max_cosine_distance = 0.75
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1) #function
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    current_date = datetime.datetime.now().date()
    count_dict = {}
  
    x,y = newLine.createLine()
    class_counter = [0,0,0,0,0] # [car,motorcycle,bus,truck,all]
    intersect_info = [] # initialise intersection list
    already_counted = deque(maxlen=50) # temporary memory for storing counted IDs
    memory = {}
    
    #รับและเก็บตำแหน่งเส้นผ่าน
    ret_val, frame = cap.read()  
    test = 1
    frameY = frame.shape[0] 
    frameX = frame.shape[1] 
    x1 = float(x[0])
    y1 = float(y[0])
    x2 = float(x[1])
    y2 = float(y[1])
    line = [(int(x1 * frameX), int(y1* frameY)), (int(x2 * frameX), int(y2 * frameY))]
    #วาดเส้นผ่าน
    cv2.line(frame, line[0], line[1], (255, 255, 255), 2)
    frameN = 0    
    while True:
        print("Frame ",frameN)
        frameN += 1
        if (test == 1):
            test = 0
        else:
            ret_val, frame = cap.read()
            #วาดเส้นผ่าน
            cv2.line(frame, line[0], line[1], (255, 255, 255), 2)
        if ret_val:
            # Process every n frames
            if mmglobal.frame_count % 3 == 0:
                outputs, img_info = predictor.inference(frame)
                if outputs == [None]:
                  args.save_result = False
                else:
                  args.save_result = True
                  #รับข้อมูลทุกอย่าง
                  result_frame, a = predictor.visual(outputs[0], img_info, predictor.confthre)
                  boxesA = a[0]
                  bbb = []
                  for bb in boxesA:
                    bx1 = float(bb[0])
                    by1 = float(bb[1])
                    bx2 = float(bb[2])
                    by2 = float(bb[3])
                    w = bx2-bx1
                    h = by2-by1
                    bbb.append([bx1,by1,w,h])
                  boxes = torch.Tensor(bbb)
                  confidence = a[1]
                  classes = a[2]
                  #ต้องdeepsortเพราะอ่านแบบเว้นเฟรม
                  features = encoder(frame, boxes)
                  # represents a bounding box detection in a single image
                  detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                                zip(boxes, confidence, classes, features)]
                  # Run non-maxima suppression.
                  boxes = np.array([d.tlwh for d in detections])        # List ของ [x y w h] ในแต่ละเฟรม
                  scores = np.array([d.confidence for d in detections]) # confidence
                  classes = np.array([d.cls for d in detections])       # class
                  indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores) #กรองเฟรมที่ซ้อนทับกันออก
                  detections = [detections[i] for i in indices]

                  # Call the tracker
                  tracker.predict()   # ได้ mean vector และ covariance matrix จาก Kalman filter prediction step
                  tracker.update(detections)

                  for track in tracker.tracks:
                      bbox = track.to_tlbr()    # (min x, miny, max x, max y)
                      track_cls = track.cls
                      cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                      if not track.is_confirmed() or track.time_since_update > 1:
                          continue

                      midpoint = track.tlbr_midpoint(bbox)
                      # get midpoint respective to botton-left
                      origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])

                      if track.track_id not in memory:
                          memory[track.track_id] = deque(maxlen=2)  

                      memory[track.track_id].append(midpoint)
                      previous_midpoint = memory[track.track_id][0]
                      origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])
                      TC = CheckCrossLine.LineCrossing(midpoint, previous_midpoint, line[0] ,line[1])
                      if TC and (track.track_id not in already_counted):
                          if (track_cls.item() == 1.0):
                            class_counter[0] += 1
                          elif (track_cls.item() == 2.0):  
                            class_counter[1] += 1
                          elif (track_cls.item() == 3.0):
                            class_counter[2] += 1
                          elif (track_cls.item() == 4.0):  
                            class_counter[3] += 1
                          class_counter[4] += 1
                          # draw alert line
                          cv2.line(frame, line[0], line[1], (0, 0, 255), 2)
                          already_counted.append(track.track_id)  # Set already counted for ID to true.
                          intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
                          intersect_info.append([track_cls, origin_midpoint, intersection_time])
                          print("class_counter[car,motorcycle,bus,truck,all] = ",class_counter)
              
                # Delete memory of old tracks.
                # This needs to be larger than the number of tracked objects in the frame.
                if len(memory) > 50:
                    del memory[list(memory)[0]]
                
                # Draw total count.
                yy = 0.1 * frame.shape[0]
                cv2.putText(frame, "Frame {}".format(str(frameN)), (int(0.05 * frame.shape[1]), int(yy)), 0,
                    1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                yy = yy + (0.1 * frame.shape[0])
                cv2.putText(frame, "Total: {}".format(str(class_counter[4])), (int(0.05 * frame.shape[1]), int(yy)), 0,
                    1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                yy = yy + (0.05 * frame.shape[0])
                cv2.putText(frame, "car: {}".format(str(class_counter[0])), (int(0.05 * frame.shape[1]), int(yy)), 0,
                    1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                yy = yy + (0.05 * frame.shape[0])
                cv2.putText(frame, "motorcycle: {}".format(str(class_counter[1])), (int(0.05 * frame.shape[1]), int(yy)), 0,
                    1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                yy = yy + (0.05 * frame.shape[0])
                cv2.putText(frame, "bus: {}".format(str(class_counter[2])), (int(0.05 * frame.shape[1]), int(yy)), 0,
                    1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                yy = yy + (0.05 * frame.shape[0])
                cv2.putText(frame, "truck: {}".format(str(class_counter[3])), (int(0.05 * frame.shape[1]), int(yy)), 0,
                    1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                    
                # Hui: Show result image
                #cv2.imshow(win_name, result_frame)

                if args.save_result:
                    vid_writer.write(result_frame)
                    vid_writer.write(frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

                mmglobal.frame_count +=1
            else:
                mmglobal.frame_count +=1
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device)
    current_time = time.localtime()
    if args.demo2 == "image":
        image_demo2(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo2 == "video" or args.demo2 == "webcam":
        imageflow_demo2(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
