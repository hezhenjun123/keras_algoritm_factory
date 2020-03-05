# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2


class image_receiver:

    def __init__(self, topic_name, image_format='bgr8'):
        self.frame_count = 0
        self.topic_name = topic_name
        # self.bridge = CvBridge()
        self.image_format = image_format

    def get_image_cv(self):
        """Get OpenCV Mat type Image.
        
        Get a Image frame with OpenCV Mat type.
        Return image alone with its timestamp and frame_id.

        Args: None
        Returns: (image, ts, frame_id)
            image: cv.Mat type(actually np.ndarray) image frame, encoding by bgr8
            ts: timestamp in millisecond
            frame_id: a string contains camera_name and frame index
        Raises: None
        """
        try:
            msg = rospy.wait_for_message(self.topic_name, Image)
        except:
            return None

        self.frame_count += 1

        if msg.encoding == 'rgba8':
            buf = np.frombuffer(msg.data, np.uint8)
            buf = buf.reshape((msg.height, msg.width, 4))
            image = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        elif msg.encoding == 'bgr8':
            buf = np.frombuffer(msg.data, np.uint8)
            image = buf.reshape((msg.height, msg.width, 3))
        else:
            rospy.loginfo('%s got invalid image, encoding:%s' % (self.topic_name, msg.encoding))
            image = None

        ts = int(msg.header.stamp.to_sec() * 1000)
        frame_id = msg.header.frame_id

        return image, ts, frame_id

    def get_image_ros(self):
        """Get ROS type Image.
        Get a image frame with ROS sensor_msgs/Image type.

        Args: None
        Returns: image
            image: sensor_msgs/Image type image frame
        Raises: None
        """
        try:
            image = rospy.wait_for_message(self.topic_name, Image)
        except:
            return None

        self.frame_count += 1

        return image
