# encoding = UTF-8

import os
import sys
import cv2


KEYPOINT_DICT = {
    '0': 'head',
    '1': 'centre',
    '2': 'right shoulder',
    '3': 'right arm',
    '4': 'right hand',
    '5': 'left shoulder',
    '6': 'left arm',
    '7': 'left hand',
    '8': 'centre hip',
    '9': 'right hip',
    '10': 'right knee',
    '11': 'right foot',
    '12': 'left hip',
    '13': 'left knee',
    '14': 'left foot',
    '15': 'nose',
    '16': 'left eye',
    '17': 'right eye',
    '18': 'left head',
    '19': 'left foot',
    '20': 'left foot',
    '21': 'left foot',
    '22': 'right foot',
    '23': 'right foot',
    '24': 'right foot'
}



class FeatureExtractor(object):
    def __init__(self, people):
        self.people = people
        
    def