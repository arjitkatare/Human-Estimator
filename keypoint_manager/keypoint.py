"""Extract frames from video using opencv and plotting keypoints from given data into given video frame and storing it as an image for reference and analysis
"""


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
        keypoints = [keypoints[i * n:(i + 1) * n] for i in range((len(keypoints) + n - 1) // n )] 
        for point in keypoints:
            temp = cv2.KeyPoint(point[0], point[1], _size = 1)
            total_keypoints.append(temp)
    cv2.drawKeypoints(image,total_keypoints, image)
    return image