# encoding = UTF-8

import os
import sys
import cv2
import numpy as np



class HumanEstimator(object):
    
    @staticmethod
    def framewise_id_getter(frame_wise_tracker, index):
        people_positions = frame_wise_tracker[str(index)]
        return people_positions
    
    def __init__(self, frame_steps = 10):
        self.data = {}
        self.frame_steps = frame_steps
        self.global_step = 0
        self.global_data = {
            'people_getting_tracked': {},
            'running_data': {}
        }
        self.people_getting_tracked = self.global_data['people_getting_tracked']
        self.step_data = None
    
    def add_people_positions_featured(self, people_positions_featured):
        self.people_positions_featured = people_positions_featured
    
    def add_featured_people(self, featured_people):
        self.featured_people = featured_people
    
    def run_people_positions(self, count):
        self.global_step += 1
        self.step_data = {} 
        for person_positions in self.people_positions_featured:
            person_id = person_positions['id']
            keypoints = person_positions['keypoints']
            zero_masker = person_positions['zero_masker']
            feature1 = person_positions['feature1']
            feature2 = person_positions['feature2']
            position = person_positions['position']
            
            # Saving a global data for re analysis purposes
            if person_id not in self.people_getting_tracked:
                self.people_getting_tracked[person_id] = [self.global_data_personId_maker(zero_masker, feature1, feature2, position, count)]
            else:
                self.people_getting_tracked[person_id].append(self.global_data_personId_maker(zero_masker, feature1, feature2, position, count))
            

            self.step_data[person_id] = self.global_data_personId_maker(zero_masker, feature1, feature2, position, count)

            
            
        self.run_process()
        
        
    def global_data_personId_maker(self, zero_masker, feature1, feature2, position, count):
        return {
            'zero_masker': zero_masker,
            'feature1': feature1,
            'feature2': feature2,
            'position': position,
            'time_step': count
        }
        
    def run_process(self):
        
        running_data =  self.global_data['running_data']
        
        
        for person_id in self.step_data:
            #Loading current step data for given person ID
            current_data = self.step_data[person_id]
            current_zero_masker = current_data['zero_masker']
            current_feature1 = current_data['feature1']
            current_feature2 = current_data['feature2'][0]
            current_position = current_data['position'][:2]
            current_timestep = current_data['time_step']
            
            # Loading running data from global data
            
            # Some initialisation in case we face boundary case
            if person_id not in running_data:
                if person_id in self.people_getting_tracked:
                    running_data[person_id] = {
                        'time_step': [],
                        'running_count': 0,
                        'position': [],
                        'feature1': [],
                        'feature2': [],
                        'zero_masker': []
                    }
                else:
                    raise "logic error"
            
            
            ###############################################################
            
            
            ### Position based estimation values calculation
            # Loading average values
            running_data_person = running_data[person_id]
            
            running_data_person['time_step'].append(current_timestep)
            running_data_person['zero_masker'].append(current_zero_masker)
            running_data_person['position'].append(current_position)
            running_data_person['feature1'].append(current_feature1)
            running_data_person['feature2'].append(current_feature2)

            running_count = running_data[person_id]['running_count']            
            running_count += 1
            running_data[person_id]['running_count'] = running_count
            
            
    def running_data_getter(self):
        return self.global_data['running_data']
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        
        
        
        