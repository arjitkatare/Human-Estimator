# encoding = UTF-8

import os
import sys
import cv2




class HumanEstimator(object):
    def __init__(self, frame_steps = 10):
        self.data = {}
        self.frame_steps = frame_steps
        self.feature1_array = []
        self.feature2_array = []
        self.global_step = 0
    
#     def add_people(self, people):
#         if self.global_step == 0:
            
        
#         self.data[person_id] = person
    
    
    def set_person(self, person):
        pass
    
    
    def run(self):
        self.global_step += 1
        
        
        
        