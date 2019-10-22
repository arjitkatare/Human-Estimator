# encoding = UTF-8

import os
import sys
import cv2
import numpy as np

# Main Estimator class handling all calculations

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
            'running_data': {},
            'human_certainity': {},
            'running_probability': {},
            'last_used_data': {},
            'analysis_started': {},
            'analysed_indexes': {}
            
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
                self.people_getting_tracked[person_id] = [self.global_data_personId_maker(
                    zero_masker, feature1, feature2, position, count)]
            else:
                self.people_getting_tracked[person_id].append(self.global_data_personId_maker(
                    zero_masker, feature1, feature2, position, count))
            

            self.step_data[person_id] = self.global_data_personId_maker(
                zero_masker, feature1, feature2, position, count)

            
            
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
                        'time_steps': [],
                        'running_count': 0,
                        'positions': [],
                        'feature1s': [],
                        'feature2s': [],
                        'zero_maskers': []
                    }
                else:
                    raise "logic error"
            
            
            ###############################################################
            
            
            ### Position based estimation values calculation
            # Loading average values
            running_data_person = running_data[person_id]
            
            running_data_person['time_steps'].append(current_timestep)
            running_data_person['zero_maskers'].append(current_zero_masker)
            running_data_person['positions'].append(current_position)
            running_data_person['feature1s'].append(current_feature1)
            running_data_person['feature2s'].append(current_feature2)

            running_count = running_data[person_id]['running_count']            
            running_count += 1
            running_data[person_id]['running_count'] = running_count
            
            
    def running_data_getter(self):
        return self.global_data['running_data']
    
    def analysis_started_getter(self):
        return self.global_data['analysis_started']
    
    def last_used_data_getter(self):
        return self.global_data['last_used_data'] 
    
    
    # This will calculate our needed average_changes for estimation of human or mannequin
    def run_calculations(self, minimum_steps = 15):
        
        running_data = self.running_data_getter()
        last_used_data = self.last_used_data_getter()
        analysis_started = self.analysis_started_getter()
        
        
        for person_id in self.people_getting_tracked:
            # Some Initialisation
            running_data_person = running_data[person_id]
            running_count = running_data_person['running_count']
            

            # Just to make sure we are not making random guesses let the data be collected upto a threshold
            if running_count > minimum_steps:
                
                # If person not yet added to the analysis routine then add it 
                if person_id not in analysis_started:
                    #Adding to analysis started tracker
                    analysis_started[person_id] = {
                        'index_done': -1
                    }
                    
                analysis_dict = analysis_started[person_id]
                
                # Loading current step data     
                current_index = analysis_dict['index_done'] + 1
                time_steps = running_data_person['time_steps']
                
                # Checking whether time_steps array is long enough
                if len(time_steps) <= current_index - 1:
                    continue
                
                positions = running_data_person['positions']
                feature1s = running_data_person['feature1s']
                feature2s = running_data_person['feature2s']
                zero_maskers = running_data_person['zero_maskers']

                # This help us in ensuring that current time steps are continuous for our average_change calculations
                # for other parameters
                
                difference_positions = self.make_positions_changes(positions) # Primary decider
                cosine_feature1s = self.make_feature1s_cosine(feature1s)  # Additionl decider, For absolute certaininty
                difference_feature1s = self.make_feature1s_changes(feature1s) # Secondary decider
                difference_feature2s = self.make_feature2s_changes(feature2s) #  Additional Secondary decider
                
                # If zero_masker are not same then we need to mask those values in calculation as the values will vary
                # if both of them are not similiar
                zero_maskers_similiarity = self.make_zero_maskers_similiarity(zero_maskers)
                difference_time_steps = self.make_time_steps_masker(time_steps, person_id)
                
                # changes calculation from here onwards
                
                
            else:
                continue
    
    def make_feature1s_changes(self, feature1s):
        feature1_zero = np.zeros_like(feature1s[0])
        right_shifted_feature1s = np.append([feature1_zero], feature1s[:-1], axis=0)
        
        difference_feature1s = -right_shifted_feature1s + feature1s
        
        difference_feature1s  =  np.sum( np.sum(difference_feature1s**2, axis = 1)**0.5, axis = 1)
        
        print('printing difference feature1s')
        print(difference_feature1s.shape, np.sum(difference_feature1s == 0) )
        return difference_feature1s
    
    
    
    def make_zero_maskers_similiarity(self, zero_maskers):
        
        previous = zero_maskers[0]
        zero_maskers_similiarity = [0]
        
        for i in zero_maskers[1:]:
            value = 1 if np.array_equal(i, previous) else 0
            zero_maskers_similiarity.append(value)
            previous = i
         
        print('printing zero_maskesr_similiarity', zero_maskers_similiarity, zero_maskers)
    
    
    def make_feature2s_changes(self, feature2s):
        right_shifted_feature2s = np.append(np.array([[0,0]]), feature2s[:-1], axis=0)
        
        difference_feature2s = -right_shifted_feature2s + feature2s
        difference_feature2s[0][0] = 0
        difference_feature2s[0][1] = 0
        
        difference_feature2s  =  np.sum(difference_feature2s**2, axis = 1)**0.5
        
        print('printing difference feature2s')
        print(difference_feature2s)
        return difference_feature2s
    
    
    @staticmethod
    def cosine_similiarity(vA,vB):
        return np.dot(vA, np.transpose(vB)) / (np.linalg.norm(vA) * np.linalg.norm(vB))
    
    def make_feature1s_cosine(self, feature1s):
        
        # Some Initialisation
        previous = feature1s[0]
        cosine_feature1s = [-1]
        for i in feature1s[1:]:
            cosine_feature1s.append(self.cosine_similiarity(i, previous)[0][0])
            previous = i 
        
        print('printing cosine features')
        print(cosine_feature1s, len(cosine_feature1s), len(feature1s))
        return np.array(cosine_feature1s)
        
    
    def make_positions_changes(self, positions):
        
        right_shifted_positions = np.append(np.array([[0,0]]), positions[:-1], axis=0)
        
        difference_positions = -right_shifted_positions + positions
        difference_positions[0][0] = 0
        difference_positions[0][1] = 0
        
        
        difference_positions  =  np.sum(difference_positions**2, axis = 1)**0.5
        
        print('printing difference positions')
        print(difference_positions)
        return difference_positions
            
    
            
    def make_time_steps_masker(self, time_steps, person_id = 'None'):
        
       
        right_shifted_time_steps = np.append(0, time_steps[:-1])

        difference_time_steps = -right_shifted_time_steps + time_steps
        difference_time_steps[0] = 0
        print('printing difference time_steps -', person_id)
        print(difference_time_steps)
        return difference_time_steps
            