

import os
import io
import cv2
import sys
import glob
import time as t
import shutil
import numpy as np
import pandas as pd
from datetime import time

import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util

from collections import namedtuple, OrderedDict
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from sys import argv

######## Frames Buffer ########

if not os.path.exists('Frames_Buffer'):
    os.mkdir('Frames_Buffer')


######## Frames creation stage ########

cam_video_dir = argv[1]
detection_cases_dir = argv[2]

for cam_vid in os.listdir(cam_video_dir): 
    count = 0
    vid_info = cam_vid.split('_')
    h,m,s = map(int, vid_info[2].split('.'))
    cam_folder = vid_info[3][3]
    vidcap = cv2.VideoCapture('Cams_Video/' + str(cam_vid))
    success = True
    if not os.path.exists('Frames_Buffer/' + str(vid_info[3][3])):
        os.mkdir('Frames_Buffer/' + str(vid_info[3][3]))
    while success:
        success, image = vidcap.read()
        if count%30 == 0 :
              if s == 59:
                m += 1
                s = 0
                frame_time = time(hour= h, minute=m, second = s).isoformat(timespec='auto')
                cv2.imwrite('Frames_Buffer/' + str(vid_info[3][3]) + '/'+'frame_'+frame_time+'_'+'.jpg', image)
              elif s < 59:
                s += 1
                frame_time = time(hour= h, minute=m, second = s).isoformat(timespec='auto')
                cv2.imwrite('Frames_Buffer/' + str(vid_info[3][3]) + '/'+'frame_'+frame_time+'_'+'.jpg', image)
        count += 1

# Grab path to current working directory
CWD_PATH = os.getcwd()
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
MODEL_NAME = 'inference_graph'
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 3

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


######## Detection stage ########


IMAGE_NAME = 'Frames_Buffer/'
analysis_time = []
for cam_folder in os.listdir(IMAGE_NAME):
    for frame in os.listdir(IMAGE_NAME + cam_folder):
      time_start = t.perf_counter()
      # Load image using OpenCV and
      # expand image dimensions to have shape: [1, None, None, 3]
      # i.e. a single-column array, where each item in the column has the pixel RGB value

      image = cv2.imread(IMAGE_NAME + cam_folder+ '/'+ frame)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image_np_expanded = np.expand_dims(image_rgb, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # Perform the actual detection by running the model with the image as input
      (boxes, scores, classes, num) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      detection_frame = vis_util.visualize_boxes_and_labels_on_image_array(image, np.squeeze(boxes),
                                                                      np.squeeze(classes).astype(np.int32),
                                                                      np.squeeze(scores),
                                                                      category_index,
                                                                      use_normalized_coordinates=True,
                                                                      line_thickness=4,
                                                                      min_score_thresh=0.70)

      objects = []
      threshold = 0.5 
      for index, value in enumerate(classes[0]):
          object_dict = {}
          if scores[0, index] > threshold:
              object_dict[category_index.get(value).get('name')] = \
                        scores[0, index]
              objects.append(object_dict)
      if 'person' in [list(i.keys())[0] for i in objects] or True in [list(i.values())[0] < 0.6 for i in objects]:
          frame_info = frame.split('_')
          a = [(k,v) for k, v in zip([list(i.keys())[0] for i in objects], [list(i.values())[0] for i in objects])]
          with open(argv[2] +'/'+'report'+'_'+frame_info[1]+'.txt','w') as text_file:
              text_file.write("Number of detected objects: {}\n".format(len(a)))
              text_file.write("Confidence, %: {}\n".format(a))
              text_file.write("Time of detection: {}\n".format(frame_info[1]))
              text_file.write("Id cam: {}".format(cam_folder))
          cv2.imwrite(argv[2] +'/' + str(cam_folder) + '_' + str(frame), detection_frame)
      time_elapsed_train = (t.perf_counter() - time_start)
      analysis_time.append(time_elapsed_train)
      print ('Successfully completed frame:', frame)
print('Average frame analysis: {} secs'.format(np.mean(analysis_time)))
shutil.rmtree('Frames_Buffer', ignore_errors=True)  