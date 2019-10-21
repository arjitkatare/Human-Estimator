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
    
    def run_people_positions(self):
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
                self.people_getting_tracked[person_id] = [self.global_data_personId_maker(zero_masker, feature1, feature2, position)]
            else:
                self.people_getting_tracked[person_id].append(self.global_data_personId_maker(zero_masker, feature1, feature2, position))
            

            self.step_data[person_id] = self.global_data_personId_maker(zero_masker, feature1, feature2, position)

            
            
        self.run_process()
        
        
    def global_data_personId_maker(self, zero_masker, feature1, feature2, position):
        return {
            'zero_masker': zero_masker,
            'feature1': feature1,
            'feature2': feature2,
            'position': position
        }
        
    def run_process(self):
        
        running_data =  self.global_data['running_data']
        
        
        for person_id in self.step_data:
            #Loading current step data for given person ID
            current_data = self.step_data[person_id]
            current_zero_masker = current_data['zero_masker']
            current_feature1 = current_data['feature1']
            current_feature2 = current_data['feature2']
            current_position = current_data['position'][:2]
            
            # Loading running data from global data
            
            # Some initialisation in case we face boundary case
            if person_id not in running_data:
                if person_id in self.people_getting_tracked:
                    running_data[person_id] = {
                        'count': 0,
                        'average_position': np.array([0,0]),
                        'average_feature1': np.array([[0,0] for _ in range(324)]),
                        'average_feature2': np.array([0,0]),
                        'position_change': [],
                        'feature1_change': [],
                        'feature2_change': []
                    }
                else:
                    raise "logic error"
                    
            ###############################################################
            ### Position based estimation values calculation
            # Loading average values
            running_data_person = running_data[person_id]
            
            average_position = running_data_person['average_position']
            average_feature1 = running_data_person['average_feature1']
            average_feature2 = running_data_person['average_feature2']
            array_position_change = running_data_person['position_change']
            array_feature1_change = running_data_person['feature1_change']
            array_feature2_change = running_data_person['feature2_change']
            count = running_data[person_id]['count']
            
            # Calculating new value            
            position_change = current_position - average_position
            
            new_average_position = average_position*count + current_position
            new_average_position = new_average_position/(count+1)
            # Updating global values
            average_position = new_average_position
            array_position_change.append(position_change)

            ### Feature1 based
            feature1_change = current_feature1[0] - average_feature1
            new_average_feature1 = average_feature1*count + current_feature1
            new_average_feature1 = new_average_feature1/(count+1)
            # Updating global values
            average_feature1 = new_average_feature1
            running_data_person['average_feature1'] = average_feature1
            array_feature1_change.append(feature1_change)
            
            ### Feature 2 based
            feature2_change = current_feature2 - average_feature2
            new_average_feature2 = average_feature2*count + current_feature2
            new_average_feature2 = new_average_feature2/(count+1)
            # Updating global values
            average_feature2 = new_average_feature2
            running_data_person['average_feature2'] = average_feature2
            array_feature2_change.append(feature2_change)

            
            count += 1
            running_data[person_id]['count'] = count 
            
            
    def running_data_getter(self):
        return self.global_data['running_data']
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        
        
        
        