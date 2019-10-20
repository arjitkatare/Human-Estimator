# encoding = UTF-8

import os
import sys
import cv2
import ndjson

from .keypoint import KeypointPrinter
from .framewiseId import FramewiseIdPrinter, FramewiseTrackerDictmaker



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
        ''' skipping frames to align with cropped videos TODO: CLearly not the best way, can be replaced by some high level wrapper for iterator but for now this will do'''
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