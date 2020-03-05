#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from ros_interface.get_image import image_receiver


def main():
    '''
    Inference ROS Node process instance example.
    
    This example show how to get image.
    In this example, image topic is passed by system argument.
    When call image_receiver.get_image_cv, process will block until receive next image frame.
    Image alone with timestamp and frame_id will be returned.
    '''
    rospy.init_node('inference_node', anonymous=True)

    if len(sys.argv) < 2:
        rospy.logerr('Please specify image topic')
        return 1

    image_topic = sys.argv[1]
    rospy.loginfo('Receive %s' % (image_topic))
    img_rcv = image_receiver(image_topic)

    save = True

    while not rospy.is_shutdown():

        ## Get OpenCV type(numpy.ndarray) image
        img, ts, frame_id = img_rcv.get_image_cv()
        if img is None:
            rospy.logerr('get image error')
            continue

        if save:
            cv2.imwrite('%d.jpg' % (ts), img)
            rospy.loginfo('%s save to %d.jpg' % (image_topic, ts))
            save = False

        rospy.loginfo('%s Got image %s' % (image_topic, frame_id))

        ## Get ROS sensor_msgs/Image type image
        # msg = img_rcv.get_image_ros()
        # rospy.loginfo('Got Image')

    rospy.loginfo('Exited.')
    return 0


if __name__ == '__main__':
    main()