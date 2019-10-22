# encoding = UTF-8

import os
import sys
import cv2
import numpy as np

KEYPOINT_DICT = {
    '0': 'nose',
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
#         np.save(open('./temp/people.pk', 'wb'), people)
        self.people_featured = None

        
    def make_features(self, feature):
        n = 3
        total_features_list = []
        if feature == 'v1':
            for i, poses in enumerate(self.people):
                keypoints = poses['pose_keypoints']
                keypoints = np.array([keypoints[i * n:((i + 1) * n)-1] for i in range((len(keypoints) + n - 1) // n )],
                                     dtype = int)
                keypoints = keypoints[:18] # Using only first 18 points as rest of them are repetetive somehow!!
                
                # Checking deactivated components and preparing them for masking
                zero_masker = np.sum(keypoints, axis = 1)
                zero_masker = zero_masker != 0
#                 print(zero_masker)
                self.dict_array_keypoints_adder(keypoints, poses, zero_masker)
                
                # Adding feature1
                # This is distance of each keypoint with all other keypoints. We are using 
                feature1 = self.feature1_extractor(keypoints, zero_masker)
                feature1 = np.reshape(feature1, (-1,2)) 
#                 print(feature1.shape)
                self.dict_array_feature_adder('feature1', feature1, poses)
                
                # Adding feature2
                # This feature is just mean of all keypoints to retain a average location of the object
                feature2 = self.feature2_extractor(keypoints)
#                 print(feature2)
                self.dict_array_feature_adder('feature2', feature2, poses)
                
                
            return self.people
                
    def dict_array_feature_adder(self, feature_name, feature, poses):
        if feature_name in poses: # This will add feature1 into the dict of self.people in loop
            poses[feature].append(feature)
        else:
            poses[feature_name] = [feature]
            
    def dict_array_keypoints_adder(self, keypoints, poses, zero_masker):
        if 'zero_masker' in poses:
            poses['zero_masker'].append(zero_masker)
            poses['keypoints'].append(keypoints)
        else:
            poses['zero_masker'] = [zero_masker]
            poses['keypoints'] = [keypoints]
            
    
    def feature1_extractor(self, keypoints, zero_masker):
        net_feature = []
        for i, keypoint in enumerate(keypoints):
            if zero_masker[i]:
                net_feature.append(keypoints - keypoint)
            else:
                net_feature.append(keypoints*0)
        return np.array(net_feature)
    
    def feature2_extractor(self, keypoints):
        feature2 = np.mean(keypoints, axis = 0)
        return np.reshape(feature2, (2,))

class FeatureExtractorForFramewise(object):
    def __init__(self, people_positions):
        self.people_positions = people_positions
        np.save(open('./temp/people_positions.pk', 'wb'), people_positions)
        

        
    def make_features(self, feature):
        n = 3
        total_features_list = []
        if feature == 'v1':
            for i, poses in enumerate(self.people_positions):
                keypoints = np.array(poses['pose']['keypoints'])
                keypoints = np.reshape(keypoints, (-1)) # Rest of the code will be handled as above class
                
                keypoints = np.array([keypoints[i * n:((i + 1) * n)-1] for i in range((len(keypoints) + n - 1) // n )],
                                     dtype = int)
#                 keypoints = keypoints[:18] # Using only first 18 points as rest of them are repetetive somehow!!
                
                # Checking deactivated components and preparing them for masking
                zero_masker = np.sum(keypoints, axis = 1)
                zero_masker = zero_masker != 0 #This will ensure that we only do calculation for same number of keypoints
#                 print(zero_masker)
                self.dict_array_keypoints_adder(keypoints, poses, zero_masker)
                
                # Adding feature1
                # This is distance of each keypoint with all other keypoints. We are using 
                feature1 = self.feature1_extractor(keypoints, zero_masker)
                feature1 = np.reshape(feature1, (-1)) 
#                 print(feature1.shape)
                self.dict_array_feature_adder('feature1', feature1, poses)
                
                # Adding feature2
                # This feature is just mean of all keypoints to retain a average location of the object
                feature2 = self.feature2_extractor(keypoints)
#                 print(feature2)
                self.dict_array_feature_adder('feature2', feature2, poses)
                
                
            return self.people_positions
                
    def dict_array_feature_adder(self, feature_name, feature, poses):
        if feature_name in poses: # This will add feature1 into the dict of self.people in loop
            poses[feature].append(feature)
        else:
            poses[feature_name] = [feature]
            
    def dict_array_keypoints_adder(self, keypoints, poses, zero_masker):
        if 'zero_masker' in poses:
            poses['zero_masker'].append(zero_masker)
            poses['keypoints'].append(keypoints)
        else:
            poses['zero_masker'] = [zero_masker]
            poses['keypoints'] = [keypoints]
            
    
    def feature1_extractor(self, keypoints, zero_masker):
        net_feature = []
        for i, keypoint in enumerate(keypoints):
            if zero_masker[i]:
                net_feature.append(keypoints - keypoint)
            else:
                net_feature.append(keypoints*0)
        return np.array(net_feature)
    
    def feature2_extractor(self, keypoints):
        feature2 = np.mean(keypoints, axis = 0)
        return np.reshape(feature2, (2,))
        
# For testing purpose we dump the initialising file and load it here to test run the class
if __name__ == '__main__':
    people = np.load('./temp/people_position.pk', allow_pickle = True)
    a = FeatureExtractorForFramewise(people)
    print(a.make_features('v1'))