# encoding = UTF-8

import os
import sys
import cv2




class HumanEstimator(object):
    def __init__(self, people):
        self.people = people
        