# encoding = UTF-8

import os
import sys
import cv2
import ndjson
import numpy as np
import pickle

from .keypoint import KeypointPrinter
from .framewiseId import FramewiseIdPrinter, FramewiseTrackerDictmaker


import human_estimator


class KeypointManager(object):
    def __init__(self, input_folder, keypoint_iterator, frame_wise_tracker, output_filepath, video_reader, video_config, start_index=None, end_index=None):
        
        self.keypoint_iterator = keypoint_iterator
        self.frame_wise_tracker = frame_wise_tracker
        self.output_filepath = output_filepath
        self.vidcap = video_reader
        self.video_config = video_config
        self.start_index = start_index
        self.end_index = end_index
        
        
        videopath = os.path.join(input_folder, 'concated.mp4')
        self.videopath = videopath
        
        self.basename, ext = os.path.splitext(os.path.basename(videopath))
        
        
    def run(self):
        #Initialisation before loop
        
        #skipping frames to align with cropped videos TODO: CLearly not the best way, 
        #can be replaced by some high level wrapper for iterator but for now this will do
        if self.start_index is not None:
            for _ in range(self.start_index):
                next(self.keypoint_iterator)
                
        keypoints_frame_data = next(self.keypoint_iterator) # Using a generator shall be prefered whenever possible
        success, image = next(self.vidcap)
        people = keypoints_frame_data['people']
        
        #Making a videowriter to make a video
        fourcc, fps, frame_size = self.video_config
        video_writer = cv2.VideoWriter(self.output_filepath, fourcc, fps, frame_size )

        count = self.start_index
        success = True
        while success:
            if success:
                if len(people) != 0:
                    image = KeypointPrinter(image, people)
                    image = FramewiseIdPrinter(image, self.frame_wise_tracker, count)

                video_writer.write(image)
            success, image = next(self.vidcap)
            keypoints_frame_data = next(self.keypoint_iterator)
            people = keypoints_frame_data['people']

            if count % 1000 == 0 :
                print('Read a new frame {}: {}'.format(count, success), end="\r")
            count += 1
            if count == self.end_index:
                success = False
        video_writer.release()
        
class KeypointManagerTest(object):
    def __init__(self, input_folder, keypoint_iterator, frame_wise_tracker, output_filepath, video_reader, video_config, start_index=None, end_index=None):
        
        self.keypoint_iterator = keypoint_iterator
        self.frame_wise_tracker = frame_wise_tracker
        self.output_filepath = output_filepath
        self.vidcap = video_reader
        self.video_config = video_config
        self.start_index = start_index
        self.end_index = end_index
        
        
        videopath = os.path.join(input_folder, 'concated.mp4')
        self.videopath = videopath
        
        self.basename, ext = os.path.splitext(os.path.basename(videopath))
        
#         pickle.save(open('./temp/keymanager.pk', 'wb'), )
        
        
    def run(self):
        #Initialisation before loop
        
        #skipping frames to align with cropped videos TODO: CLearly not the best way, 
        #can be replaced by some high level wrapper for iterator but for now this will do
        if self.start_index is not None:
            for _ in range(self.start_index):
                next(self.keypoint_iterator)
                
        keypoints_frame_data = next(self.keypoint_iterator) # Using a generator shall be prefered whenever possible
        success, image = next(self.vidcap)
        people = keypoints_frame_data['people']
        
        #Making a videowriter to make a video
        fourcc, fps, frame_size = self.video_config
        video_writer = cv2.VideoWriter(self.output_filepath, fourcc, fps, frame_size )

        count = self.start_index
        success = True
        
        estimator = human_estimator.HumanEstimator()
        
        while success:
            features = None
            featured_people = None
            people_positions = None
            if success:
                if len(people) != 0 and str(count) in self.frame_wise_tracker:
                    
                    print('number of people = ' + str(len(people)))
                    print('Making Features')
                    features = human_estimator.FeatureExtractor(people)
                    featured_people = features.make_features('v1')
#                     print(featured_people)
                    
                    image = KeypointPrinter(image, people)
                    image = FramewiseIdPrinter(image, self.frame_wise_tracker, count)
                    
                    # Using static method from estimator
                    people_positions = estimator.framewise_id_getter(self.frame_wise_tracker, count) 
                    
                    people_positions_features = human_estimator.FeatureExtractorForFramewise(people_positions)
                    people_positions_featured = people_positions_features.make_features('v1')
#                     print('Prining people position featured')
#                     print(people_positions_featured)
                    
                    estimator.add_people_positions_featured(people_positions_featured)
                    estimator.add_featured_people(featured_people)
                    estimator.run_people_positions(count)
                    
                    print('printin keys of running_data')
                    print(estimator.running_data_getter().keys())
                    print('printing positions array')
                    print(estimator.running_data_getter()['7275']['positions'])
                    print('printing feature2s array')
                    print(estimator.running_data_getter()['7275']['feature2s'])
                    print('printing running count')
                    print(estimator.running_data_getter()['7275']['running_count'])
                    print('printing time steps')
                    print(estimator.running_data_getter()['7275']['time_steps'])
                    print(len(estimator.running_data_getter()))
                    
                    #Starting Calculation
                    estimator.run_calculations()
                    
                # Writing Image
                video_writer.write(image)
            success, image = next(self.vidcap)
            keypoints_frame_data = next(self.keypoint_iterator)
            people = keypoints_frame_data['people']

            if count % 1000 == 0 :
                print('Read a new frame {}: {}'.format(count, success), end="\r")
            count += 1
            if count == self.end_index:
                success = False
            print('###################')
            print(count)
            input()
        video_writer.release()
        
if __name__ == '__main__':
    
    input_folder, keypoint_iterator, frame_wise_tracker, output_filepath, video_reader, video_config, start_index, end_index = pickle.load(open('./temp/keymanager.pk', 'rb'))
    manager = keypoint_manager.KeypointManagerTest(input_folder, keypoint_iterator, frame_wise_tracker, output_filepath, video_reader, video_config, start_index, end_index)
    print('started running manager')
    manager.run()