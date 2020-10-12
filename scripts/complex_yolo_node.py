#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2 
from complex_yolo_ros.msg import DetectedObjectArray, DetectedObject
import ros_numpy

import numpy as np
import math, os, argparse, time
import cv2
import torch
from skimage.transform import resize
import threading
from time import sleep

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
        print("Init()")
        rospy.init_node('complex_yolo_node', anonymous = True)

        # Init lock variable
        self.lock = threading.Lock()

        self.lidar_topic = rospy.get_param("~lidar_topic")	

        sub_lidar = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidarCb, queue_size=1)

        self.model_def = rospy.get_param("~model_def")
        self.weights_path = rospy.get_param('~weights_path')
        self.class_path = rospy.get_param('~class_path')
        self.conf_thres = rospy.get_param('~conf_thres')
        self.nms_thres = rospy.get_param('~nms_thres')
        self.img_size = rospy.get_param('~img_size', cnf.BEV_WIDTH)
        self.save_video = rospy.get_param('~save_video', False)

        print("Info")
        print("  Config path: {0:s}".format(self.model_def))
        print("  Weights path {0:s}".format(self.weights_path))
        print("  Class path: {0:s}".format(self.class_path))
        print("  Confidence threshold: {0:.2f}".format(self.conf_thres))
        print("  NMS threshold: {0:.2f}".format(self.nms_thres))
			
        self.classes = utils.load_classes(self.class_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lidar data status
        self.lidar_status = False

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # Set up model
        self.model = Darknet(self.model_def, img_size = self.img_size).to(self.device)
        # Load checkpoint weights
        self.model.load_state_dict(torch.load(self.weights_path))
        # Eval mode
        self.model.eval()

        # Load dataset
#		self.dataset = KittiYOLO2WayDataset(cnf.root_dir, split=self.split, folder=self.folder)
#		self.data_loader = torch_data.DataLoader(self.dataset, 1, shuffle=False)

    def lidarCb(self, data):
        np_lidar = self.pointcloud2_to_array(data)

        # Lock
        self.lock.acquire()

        front_lidar = bev_utils.removePoints(np_lidar, cnf.boundary)
        self.front_bevs = bev_utils.makeBVFeature(front_lidar, cnf.DISCRETIZATION, cnf.boundary)
        self.front_bevs = self.front_bevs.reshape(1,3,608,608)

#		self.front_bevs = torch.from_numpy(self.front_bevs).float()
#		self.front_bevs = self.front_bevs[None]

#		print(self.front_bevs)
#		imgs = Variable(self.front_bevs.type(self.Tensor))
#		print(imgs)

        back_lidar = bev_utils.removePoints(np_lidar, cnf.boundary_back)
        self.back_bevs = bev_utils.makeBVFeature(back_lidar, cnf.DISCRETIZATION, cnf.boundary_back)
        self.back_bevs = self.back_bevs.reshape(1,3,608,608)
#		bev_maps = np.flip(self.back_bevs, [2,3])
#		bev_maps = bev_maps.reshape(3,608,608)

#		self.back_bevs = torch.from_numpy(self.back_bevs).float()
#
#		print(self.back_bevs)
#		imgs = Variable(self.back_bevs.type(self.Tensor))
#		print(imgs)
		
        # Subscribe Lidar data
        self.lidar_status = True

        # Unlock
        self.lock.release()

    # Pointcloud to numpy array (x,y,z,intensity)
    def pointcloud2_to_array(self, cloud_msg, squeeze = True):
        DUMMY_FIELD_PREFIX = '__'
        dtype_list = ros_numpy.point_cloud2.fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

        cloud_arr = np.fromstring(cloud_msg.data, dtype_list)

        cloud_arr = cloud_arr[[fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

        cloud_arr_x = cloud_arr['x']
        cloud_arr_y = cloud_arr['y']
        cloud_arr_z = cloud_arr['z']
        cloud_arr_intensity = cloud_arr['intensity']

        cloud_ = np.vstack([cloud_arr_x, cloud_arr_y, cloud_arr_z, cloud_arr_intensity])
	
        return cloud_.T
	
    def detect_and_draw(self, model, bev_maps, Tensor, is_front = True):

        # Numpy to torch
        bev_maps = torch.from_numpy(bev_maps).float()

        imgs = Variable(bev_maps.type(Tensor))

        if not is_front:
            bev_maps = torch.flip(bev_maps, [2, 3])

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

        Imap = display_bev[:, :, 2]
        Hmap = display_bev[:, :, 1]
        Dmap = display_bev[:, :, 0]

        raw_bev = display_bev

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

        return display_bev, img_detections, Hmap, Imap, Dmap, raw_bev

    def run(self):
        print("Run detector()")
        if self.save_video:
            out = cv2.VideoWriter('bev_detection_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (self.img_size*2, self.img_size+375))

        # Check status
        while not self.lidar_status:
            print("Waiting for {0:s} lidar data...".format(self.lidar_topic))
            sleep(2)

        start_time = time.time()
		
        while not rospy.is_shutdown():
#			img2d = cv2.imread(img_path)

#			self.input_img = self.imagePreProcessing(img2d)

            # Lock 
            self.lock.acquire()

            front_bev_result, front_detections, front_Hmap, front_Imap, front_Dmap, front_bev_raw = self.detect_and_draw(self.model, self.front_bevs, self.Tensor, True)
            back_bev_result, back_detections, back_Hmap, back_Imap, back_Dmap, back_bev_raw = self.detect_and_draw(self.model, self.back_bevs, self.Tensor, False)

            # Unlock
            self.lock.release()

            front_bev_result_eval = front_bev_result
            back_bev_result_eval = back_bev_result

            end_time = time.time()
            print(f"FPS: {1.0/(end_time-start_time):.2f}")
            start_time = end_time

            front_bev_result = cv2.rotate(front_bev_result, cv2.ROTATE_90_CLOCKWISE)
            back_bev_result = cv2.rotate(back_bev_result, cv2.ROTATE_90_COUNTERCLOCKWISE)

            front_bev_raw = cv2.rotate(front_bev_raw, cv2.ROTATE_90_CLOCKWISE)
            back_bev_raw = cv2.rotate(back_bev_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
				
            vis = np.concatenate((front_bev_result, back_bev_result), axis=1)
            vis_raw = np.concatenate((front_bev_raw, back_bev_raw), axis=1)
            vis_eval = np.concatenate((front_bev_result_eval, back_bev_result_eval), axis=1)

#			calib = kitti_utils.Calibration(img_paths[0].replace(".jpg", ".txt").replace("image_2", "calib"))
#			objects_pred = predictions_to_kitti_format(front_detections, calib, img2d.shape, self.img_size)  
#img2d = mview.show_image_with_boxes(img2d, objects_pred, calib, False)

#			img2d = cv2.resize(img2d, (self.img_size*2, 375))
#			vis = np.concatenate((img2d, vis), axis=0)

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
        run_thread = threading.Thread(target = complex.run())
        rospy.spin()
    except rospy.ROSInterruptException:
        threads.join()
        rospy.shutdown()
