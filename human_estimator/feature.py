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
                keypoints = np.array([keypoints[i * n:((i + 1) * n)-1] for i in range((len(keypoints) + n - 1) // n )], dtype = int)
                keypoints = keypoints[:18]
                
                # Checking deactivated components and preparing them for masking
                zero_masker = np.sum(keypoints, axis = 1)
                zero_masker = zero_masker != 0
                print(zero_masker)
                
                feature1 = self.feature1_extractor(keypoints, zero_masker)
                feature1 = np.reshape(feature1, (-1))
                
                print(feature1.shape)
                
                if 'feature1' in poses:
                    poses['feature1'].append(feature1)
                    poses['zero_masker'].append(zero_masker)
                else:
                    poses['feature1'] = [feature1]
                    poses['zero_masker'] = [zero_masker]
                
            return self.people
                
                
    def feature1_extractor(self, keypoints, zero_masker):
        net_feature = []
        for i, keypoint in enumerate(keypoints):
            if zero_masker[i]:
                net_feature.append(keypoints - keypoint)
            else:
                net_feature.append(keypoints*0)
        return np.array(net_feature)
                

        
# For testing purpose we dump the initialising file and load it here to test run the class
if __name__ == '__main__':
    people = np.load('./temp/people.pk', allow_pickle = True)
    a = FeatureExtractor(people)
    print(a.make_features('v1'))