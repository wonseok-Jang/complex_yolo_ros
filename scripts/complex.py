#! /usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2 
from cv_bridge import CvBridge, CvBridgeError
from complex_yolo_ros.msg import DetectedObjectArray, DetectedObject

import numpy as np
import math, os, argparse, time
import cv2
import torch

from models import *

import utils.utils as utils
import torch.utils.data as torch_data
from utils.kitti_yolo_dataset import KittiYOLO2WayDataset
import utils.kitti_bev_utils as bev_utils
import utils.kitti_utils as kitti_utils
import utils.config as cnf
from test_detection import predictions_to_kitti_format


class ComplexYOLO:
	def __init__(self):
		rospy.init_node('complex_yolo_node', anonymous = True)

		# Init Cv bridge
		self.bridge = CvBridge()

		self.image_topic = rospy.get_param("~image_topic")	
		self.lidar_topic = rospy.get_param("~lidar_topic")	

		sub_image = rospy.Subscriber(self.image_topic, Image, self.img_callback, queue_size=1)
#		sub_lidar = rospy.Subscriber(self.lidar_topic, PointCloud2, sefl.lidar_callback, queue_size=1)

		self.model_def = rospy.get_param("~model_def")
		self.weights_path = rospy.get_param('~weights_path')
		self.class_path = rospy.get_param('~class_path')
		self.conf_thres = rospy.get_param('~conf_thres')
		self.nms_thres = rospy.get_param('~nms_thres')
		self.img_size = rospy.get_param('~img_size', cnf.BEV_WIDTH)
		self.save_video = rospy.get_param('~save_video', False)
		self.split = rospy.get_param('~split')
		self.folder = rospy.get_param('~folder')
			
		self.classes = utils.load_classes(self.class_path)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up model
#        self.model = Darknet('/aveesSSD/AES/ws_aes/src/complex_yolo/config/complex_yolov3.cfg', img_size = self.img_size).to(self.device)
		self.model = Darknet(self.model_def, img_size = self.img_size).to(self.device)
#    	# Load checkpoint weights
		self.model.load_state_dict(torch.load(self.weights_path))
#    	# Eval mode
		self.model.eval()

		self.dataset = KittiYOLO2WayDataset(cnf.root_dir, split=self.split, folder=self.folder)
		self.data_loader = torch_data.DataLoader(self.dataset, 1, shuffle=False)

		self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

	def img_callback(self, data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
		except CvBridgeError as e:
			print(e)

	def detect_and_draw(self, model, bev_maps, Tensor, is_front = True):
		if not is_front:
			bev_maps = torch.flip(bev_maps, [2, 3])
		imgs = Variable(bev_maps.type(Tensor))

		img_detections = []
		with torch.no_grad():
			detections = self.model(imgs)
			detections = utils.non_max_suppression_rotated_bbox(detections, self.conf_thres, self.nms_thres)

		img_detections.extend(detections)

    	# Only supports single batch
		display_bev = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
    
		bev_map = bev_maps[0].numpy()
		display_bev[:, :, 2] = bev_map[0, :, :]  # r_map
		display_bev[:, :, 1] = bev_map[1, :, :]  # g_map
		display_bev[:, :, 0] = bev_map[2, :, :]  # b_map

		display_bev *= 255
		display_bev = display_bev.astype(np.uint8)

		for detections in img_detections:
			if detections is None:
				continue
        	# Rescale boxes to original image
			detections = utils.rescale_boxes(detections, self.img_size, display_bev.shape[:2])
			for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
				
				yaw = np.arctan2(im, re)
            	# Draw rotated box
				bev_utils.drawRotatedBox(display_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

		return display_bev, img_detections

	def run(self):
		if self.save_video:
			out = cv2.VideoWriter('bev_detection_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (self.img_size*2, self.img_size+375))

		start_time = time.time()
		
		while not rospy.is_shutdown():
			for index, (img_paths, front_bevs, back_bevs) in enumerate(self.data_loader):

				front_bev_result, img_detections = self.detect_and_draw(self.model, front_bevs, self.Tensor, True)
				back_bev_result, _ = self.detect_and_draw(self.model, back_bevs, self.Tensor, False)

				end_time = time.time()
				print(f"FPS: {1.0/(end_time-start_time):.2f}")
				start_time = end_time

				front_bev_result = cv2.rotate(front_bev_result, cv2.ROTATE_90_CLOCKWISE)
				back_bev_result = cv2.rotate(back_bev_result, cv2.ROTATE_90_COUNTERCLOCKWISE)
				vis = np.concatenate((front_bev_result, back_bev_result), axis=1)

				img2d = cv2.imread(img_paths[0])
#img2d = cv2.imread("/aveesSSD/jang/Complex-YOLOv3/data/KITTI/object/image_2/000001.jpg")

				calib = kitti_utils.Calibration(img_paths[0].replace(".jpg", ".txt").replace("image_2", "calib"))
				objects_pred = predictions_to_kitti_format(img_detections, calib, img2d.shape, self.img_size)  
#img2d = mview.show_image_with_boxes(img2d, objects_pred, calib, False)

				img2d = cv2.resize(img2d, (self.img_size*2, 375))
				vis = np.concatenate((img2d, vis), axis=0)

				cv2.imshow('BEV_DETECTION_RESULT', vis)
				if self.save_video:
					out.write(vis)

				if cv2.waitKey(1) & 0xFF == 27:
					break

			if self.save_video:
				out.release()

if __name__ == '__main__':
	try:
		complex = ComplexYOLO()
		complex.run()
#rospy.spin()
		r = rospy.Rate(1000)
		r.sleep()
#complex.sum()
	except rospy.ROSInterruptException:
		rospy.shutdown()
