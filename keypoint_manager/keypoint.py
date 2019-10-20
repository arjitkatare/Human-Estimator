# encoding = UTF-8

import os
import sys
import cv2
import ndjson

def KeypointPrinter(image, people):
    
    total_keypoints = []
    chunk_size = 3
    n = chunk_size
    for poses in people:
        keypoints = poses['pose_keypoints']
        keypoints = [keypoints[i * n:(i + 1) * n] for i in range((len(keypoints) + n - 1) // n )] #Splitting array into array of subarrays of size n which is 3 for this case
        for i, point in enumerate(keypoints):
            if i != 0:
                continue
            temp = cv2.KeyPoint(point[0], point[1], _size = 10)
            total_keypoints.append(temp)
    cv2.drawKeypoints(image,total_keypoints, image)
    return image