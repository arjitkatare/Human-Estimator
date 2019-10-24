# encoding = UTF-8

import os
import sys
import cv2
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

# Main Estimator class handling all calculations

# This help us in stopping all those print statemetns while executing process
check = 2
def printed():
    if check == 1:
        return print
    else:
        def dummy(a = None, b = None, c = None, d = None, e = None, f = None, end = None):
            pass
        return dummy

print = printed()


class OptimizedHumanEstimator(object):
    
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
        self.running_probability = self.global_data['running_probability']
        self.people_getting_tracked = self.global_data['people_getting_tracked']
        self.human_certainity = self.global_data['human_certainity']
        self.running_data = self.global_data['running_data']
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
        people_getting_tracked = self.people_getting_tracked_getter()
        Parallel(n_jobs=num_cores)(delayed(self.run_calculations)(i) for i in people_getting_tracked)
        
        
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
            if person_id in self.human_certainity:
                self.running_data
                continue
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
            
    
    def people_getting_tracked_getter(self):
        return self.global_data['people_getting_tracked'].keys()
    
    def running_data_getter(self):
        return self.global_data['running_data']
    
    def analysis_started_getter(self):
        return self.global_data['analysis_started']
    
    def last_used_data_getter(self):
        return self.global_data['last_used_data'] 
    
    
    # This will calculate our needed average_changes for estimation of human or mannequin
    def run_calculations(self, person_id, minimum_steps = 5):
        
        running_data = self.running_data_getter()
        last_used_data = self.last_used_data_getter()
        analysis_started = self.analysis_started_getter()

            # This will free some memory
        if person_id in self.human_certainity:
            self.running_data[person_id] = None
            return



        # Some Initialisation
        running_data_person = running_data[person_id]
        running_count = running_data_person['running_count']


        # Just to make sure we are not making random guesses let the data be collected upto a threshold
        if running_count > minimum_steps:


            # Loading current step data     
            time_steps = running_data_person['time_steps']



            positions = running_data_person['positions']
            feature1s = running_data_person['feature1s']
            feature2s = running_data_person['feature2s']
            zero_maskers = running_data_person['zero_maskers']

            # This help us in ensuring that current time steps are continuous for our average_change calculations
            # for other parameters
            difference_time_steps = self.make_time_steps_masker(time_steps, person_id)                
            difference_positions = self.make_positions_changes(positions) # Primary decider
            cosine_feature1s = self.make_feature1s_cosine(feature1s)  # Additionl decider, For absolute certaininty
            difference_feature1s = self.make_feature1s_changes(feature1s) # Secondary decider
            difference_feature2s = self.make_feature2s_changes(feature2s) #  Additional Secondary decider

            # If zero_masker are not same then we need to mask those values in calculation as the values will vary
            # if both of them are not similiar
            zero_maskers_similiarity = self.make_zero_maskers_similiarity(zero_maskers) # Even this is a decider as continuos change in available keypoints show movements


            final_mask = difference_time_steps * zero_maskers_similiarity
            print(final_mask)
            # changes calculation from here onwards

            # Voting
            self.election_commission(person_id, difference_feature2s, final_mask, 2, 5)
            self.election_commission(person_id, difference_feature1s, final_mask, 1, 5)
            self.election_commission(person_id, cosine_feature1s, final_mask, 100, 100)
            self.election_commission(person_id, difference_positions, final_mask, 100, 100)
            self.election_commission_zero_masker_similiarity(person_id, zero_maskers_similiarity, 0.7)
#                 self.election_commission(person_id, difference_feature1s)
            print(person_id)

    
    def election_commission_zero_masker_similiarity(self, person_id, zero_maskers_similiarity, threshold):
        print('zero_masker_similiarity', zero_maskers_similiarity)
        zero_counter = np.array(zero_maskers_similiarity) == 0
        print(zero_counter)
        probability = float(np.sum(zero_counter))/ len(zero_maskers_similiarity)
        print('printing probability zero masker - ', probability)
        
        if person_id not in self.running_probability:
            self.running_probability[person_id] = probability
        else:
        
            if probability > threshold and len(zero_maskers_similiarity) > 10:
                self.running_probability[person_id] = (self.running_probability[person_id] + probability) * 0.5
                self.human_certainity[person_id] = 1
    
    def election_commission(self, person_id, difference, final_mask, threshold = 2, vote_threshold = 5):
        
        jumps = self.jump_detector(difference, final_mask, threshold)
        print(jumps)
        vote = np.sum(jumps)
        probability = vote/vote_threshold
        if person_id not in self.running_probability:
            self.running_probability[person_id] = probability
        else:
            self.running_probability[person_id] = (self.running_probability[person_id] + probability) * 0.5
        
        if vote > vote_threshold:
            self.human_certainity[person_id] = 1
    
    
    
    @staticmethod
    def jump_detector(array, final_mask, threshold):
        mask_array = array * final_mask != 0
        usable_array = array[mask_array]
        print('printing usable array')
        print(usable_array)
        
        jumps = usable_array >= threshold
        
        return jumps
        
            
            
                
    
    
    def make_feature1s_changes(self, feature1s):
        feature1_zero = np.zeros_like(feature1s[0])
        right_shifted_feature1s = np.append([feature1_zero], feature1s[:-1], axis=0)
        
        difference_feature1s = -right_shifted_feature1s + feature1s
        
        difference_feature1s  =  np.sum( np.sum(difference_feature1s**2, axis = 1)**0.5, axis = 1)/648.0
        
        print('printing difference feature1s')
        print(difference_feature1s.shape, difference_feature1s)
        return difference_feature1s
    
    
    
    def make_zero_maskers_similiarity(self, zero_maskers):
        
        previous = zero_maskers[0]
        zero_maskers_similiarity = [0]
        
        for i in zero_maskers[1:]:
            value = 1 if np.array_equal(i, previous) else 0
            zero_maskers_similiarity.append(value)
            previous = i
         
        print('printing zero_maskesr_similiarity', zero_maskers_similiarity, zero_maskers)
        return zero_maskers_similiarity
    
    
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
        difference_time_steps = difference_time_steps == 1
        print('printing difference time_steps -', person_id)
        print(difference_time_steps)
        return difference_time_steps
            