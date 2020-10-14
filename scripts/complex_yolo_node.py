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
import threading as thread
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
#        print("Init()")
        rospy.init_node('complex_yolo_node', anonymous = True)

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
        
        self.complex_yolo_init()
			
    def complex_yolo_init(self):
        # Init lock variable
        self.lock = thread.Lock()

        self.lidar_topic = rospy.get_param("~lidar_topic")	
        self.detected_objects_topic = rospy.get_param('~detected_objects_topic')

        sub_lidar = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidarCb, queue_size=1)
        self.pub_dets = rospy.Publisher(self.detected_objects_topic, DetectedObjectArray, queue_size=1)


        self.classes = utils.load_classes(self.class_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lidar data status
        self.lidar_status = False
        self.np_lidar = 0

        # Measurement
        self.max_iter = 140
        self.num_drop = 4
        self.count = 0 # loop counter
        self.d_conv = np.zeros(self.max_iter)
        self.d_infer = np.zeros(self.max_iter)
        self.d_proc = np.zeros(self.max_iter)
        self.e2e_delay = np.zeros(self.max_iter)
        self.cycle_time = np.zeros(self.max_iter)

        # Timestamp
        self.recv_timestamp = 0
        self.data_timestamp = np.zeros(3)
        self.timestamp_index = 0

        # Detection result
        self.front_bev_result = 0
        self.back_bev_result = 0
        self.front_bev_raw = 0
        self.back_bev_raw = 0

        # Init detection id
        self.classes_h = [1.6, 1.7, 1.7] # Car, Pedestrian, Cyclist

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # Set up model
        self.model_f = Darknet(self.model_def, img_size = self.img_size).to(self.device)
        self.model_b = Darknet(self.model_def, img_size = self.img_size).to(self.device)
        # Load checkpoint weights
        self.model_f.load_state_dict(torch.load(self.weights_path))
        self.model_b.load_state_dict(torch.load(self.weights_path))
        # Eval mode
        self.model_f.eval()
        self.model_b.eval()

    def lidarCb(self, data):
        convert_lidar = self.pointcloud2_to_array(data)

#        # Lock
        self.lock.acquire()

        self.recv_timestamp = time.monotonic()*1000.
        self.np_lidar = convert_lidar

#        # Unlock
        self.lock.release()
        
        # Subscriber status and receive timestamp
        self.lidar_status = True

    def lidar_to_bev(self, lidar_data, lidar_time):
        convert_start = time.monotonic()
        convert_index = self.timestamp_index

        self.data_timestamp[convert_index] = lidar_time

        front_lidar = bev_utils.removePoints(lidar_data, cnf.boundary)
        front_bev = bev_utils.makeBVFeature(front_lidar, cnf.DISCRETIZATION, cnf.boundary)
        front_bev = front_bev.reshape(1,3,608,608)

#       self.front_bevs = torch.from_numpy(self.front_bevs).float()
#       self.front_bevs = self.front_bevs[None]

#       print(self.front_bevs)
#       imgs = Variable(self.front_bevs.type(self.Tensor))
#       print(imgs)

        back_lidar = bev_utils.removePoints(lidar_data, cnf.boundary_back)
        back_bev = bev_utils.makeBVFeature(back_lidar, cnf.DISCRETIZATION, cnf.boundary_back)
        back_bev = back_bev.reshape(1,3,608,608)

#       bev_maps = np.flip(self.back_bevs, [2,3])
#       bev_maps = bev_maps.reshape(3,608,608)

#       self.back_bevs = torch.from_numpy(self.back_bevs).float()
#
#       print(self.back_bevs)
#       imgs = Variable(self.back_bevs.type(self.Tensor))
#       print(imgs)

#        self.tmp_front_bevs = front_bev
#        self.tmp_back_bevs = back_bev

        convert_end = time.monotonic()
        
        if self.count >= self.num_drop:
            self.d_conv[self.count-self.num_drop] = (convert_end - convert_start)*1000.
            print("Converting: {0:0.2f}".format(self.d_conv[self.count-self.num_drop]))

        return front_bev, back_bev

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

    # Draw detections and make messages
    def make_publish_msg(self, img_detections, bev):
        for detections in img_detections:
            if detections is None:
                continue
            # Rescale boxes to original image
            detections = utils.rescale_boxes(detections, self.img_size, bev.shape[:2])
            for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
                yaw = np.arctan2(im, re)
                h = self.classes_h[int(cls_pred)]

                # Draw rotated box
                bev_utils.drawRotatedBox(bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

                detection_msg = DetectedObject()
                detection_msg.pose.position.x = x
                detection_msg.pose.position.y = y 
                detection_msg.pose.position.z = h/2. 

                detection_msg.pose.orientation.x = .0
                detection_msg.pose.orientation.y = .0
                detection_msg.pose.orientation.z = yaw 
                detection_msg.pose.orientation.w = 1. 

                detection_msg.dimensions.x = w
                detection_msg.dimensions.y = l 
                detection_msg.dimensions.z = h 

                detection_msg.label = self.classes[int(cls_pred)] 
                detection_msg.score = conf
                detection_msg.id = self.dets_id

                self.detection_results.objects.append(detection_msg)
                self.dets_id += 1

    def post_processing(self, f_bev_result, b_bev_result, f_bev_raw, b_bev_raw, f_detections, b_detections):
        processing_start = time.monotonic()
        post_processing_index = self.timestamp_index

        # Make ros msg
        self.make_publish_msg(f_detections, f_bev_result)
        self.make_publish_msg(b_detections, b_bev_result)

        # Publish data
        self.pub_dets.publish(self.detection_results)            

        if self.count > self.num_drop:
            self.e2e_delay[self.count-self.num_drop] = time.monotonic()*1000. - self.data_timestamp[post_processing_index]
            print("Delay: {0:0.2f}".format(self.e2e_delay[self.count-self.num_drop]))

        # Merge front bev with back bev
        f_bev_result_eval = f_bev_result
        b_bev_result_eval = b_bev_result

        f_bev_result = cv2.rotate(f_bev_result, cv2.ROTATE_90_CLOCKWISE)
        b_bev_result = cv2.rotate(b_bev_result, cv2.ROTATE_90_COUNTERCLOCKWISE)

        f_bev_raw = cv2.rotate(f_bev_raw, cv2.ROTATE_90_CLOCKWISE)
        b_bev_raw = cv2.rotate(b_bev_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
				
        self.vis = np.concatenate((f_bev_result, b_bev_result), axis=1)
        vis_raw = np.concatenate((f_bev_raw, b_bev_raw), axis=1)
        vis_eval = np.concatenate((f_bev_result_eval, b_bev_result_eval), axis=1)

        processing_end = time.monotonic()
        
        if self.count > self.num_drop:
            self.d_proc[self.count-self.num_drop] = (processing_end - processing_start)*1000.
            print("Post processing: {0:0.2f}".format(self.d_proc[self.count-self.num_drop]))


    def inference(self, model, bev_maps, Tensor, is_front = True):
        infer_start = time.monotonic()
        # Numpy to torch
        bev_maps = torch.from_numpy(bev_maps).float()

        imgs = Variable(bev_maps.type(Tensor))

        if not is_front:
            bev_maps = torch.flip(bev_maps, [2, 3])

        img_detections = []
        with torch.no_grad():
            detections = model(imgs)
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

        if is_front:
            self.front_bev_result = display_bev
            self.front_bev_raw = raw_bev
            self.front_dets = img_detections
        else:
            self.back_bev_result = display_bev
            self.back_bev_raw = raw_bev
            self.back_dets = img_detections

        infer_end = time.monotonic() 
        
        if self.count > self.num_drop:
            self.d_infer[self.count-self.num_drop] = (infer_end - infer_start) * 1000.
            print("Inference : {0:0.2f}".format(self.d_infer[self.count-self.num_drop]))

#        return display_bev, img_detections, Hmap, Imap, Dmap, raw_bev

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
            print("======================")
            # Init msg & id
            self.detection_results = DetectedObjectArray()
            self.dets_id = 0

#            f_infer_thread = thread.Thread(target = self.inference, args=(self.model_f, self.front_bevs, self.Tensor, True))   # Front bev inference thread
#            b_infer_thread = thread.Thread(target = self.inference, args=(self.model_b, self.back_bevs, self.Tensor, False))   # Back bev inference thread
            # Post processing (publish msg, merge image)
#            post_processing_thread = thread.Thread(target = self.post_processing, args=(local_front_bev_result, local_back_bev_result, local_front_bev_raw, local_back_bev_raw, local_front_dets, local_back_dets))

#            f_infer_thread.start()
#            b_infer_thread.start()
#            post_processing_thread.start()

            ### Main thread
            # Lock
            self.lock.acquire()

            tmp_front_bevs, tmp_back_bevs = self.lidar_to_bev(self.np_lidar, self.recv_timestamp)
            self.front_bevs = tmp_front_bevs
            self.back_bevs = tmp_back_bevs
           
            # Unlock
            self.lock.release()

            self.inference(self.model_f, self.front_bevs, self.Tensor, True)
            self.inference(self.model_b, self.back_bevs, self.Tensor, False)
            
            self.post_processing(self.front_bev_result, self.back_bev_result, self.front_bev_raw, self.back_bev_raw, self.front_dets, self.back_dets)
            ### Main thread end

            end_time = time.time()
            print(f"FPS: {1.0/(end_time-start_time):.2f}")
            print(f"Cycle time: {(end_time-start_time) * 1000.:.2f}")

            if self.count >= self.num_drop:
                self.cycle_time[self.count-self.num_drop] = (end_time-start_time) * 1000.

            start_time = end_time

            cv2.imshow('BEV_DETECTION_RESULT', self.vis)

            if self.save_video:
                out.write(vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            self.count += 1
            print("Count: {0:d}".format(self.count))

            if self.count >= (self.max_iter+self.num_drop):
                print("===========================")
                print("Converting delay (ms): {0:0.2f}".format(np.mean(self.d_conv)))
                print("Inference delay (ms): {0:0.2f}".format(np.mean(self.d_infer)))
                print("Post-processing delay (ms): {0:0.2f}".format(np.mean(self.d_proc)))
                print("End-to-end delay (ms): {0:0.2f}".format(np.mean(self.e2e_delay)))
                print("Cycle time (ms): {0:0.2f}".format(np.mean(self.cycle_time)))
                break

        if self.save_video:
            out.release()

if __name__ == '__main__':
    try:
        complex_yolo = ComplexYOLO()
        run_thread = thread.Thread(target = complex_yolo.run)

        run_thread.start()
        rospy.spin()
        run_thread.join()
    except rospy.ROSInterruptException:
        run_thread.join()
