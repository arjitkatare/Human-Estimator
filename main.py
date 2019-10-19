# encoding = UTF-8

import os
import sys
import cv2
import ndjson
from optparse import OptionParser


# Custom imports
from video_extractor.videoExtractor import VideoExtractor
from keypoint_manager.keypoint import KeypointPrinter
from keypoint_manager.framewiseId import FramewiseIdPrinter, FramewiseTrackerDictmaker 



def main():
    
    parser = OptionParser()
    
    
    parser.add_option("-d", "--data-folder",
                        dest = "input_folder",
                        help = "Add datafolder path",
                        type = "string",
                        action = "store"
                        )
    parser.add_option("-f", "--filename",
                        dest = "filename",
                        help = "Add Filename",
                        type = "string",
                        action = "store"
                        )
    parser.add_option("-o", "--output-folder",
                        dest = "output_folder",
                        help = "Add ouputfolder path",
                        type = "string",
                        action = "store"
                        )
    parser.add_option("-s", "--start-index",
                        dest = "start_index",
                        help = "Add start index for the video frame you want to extract",
                        type = "int",
                        action = "store"
                        )
    parser.add_option("-e", "--end-index",
                        dest = "end_index",
                        help = "Add end index",
                        type = "int",
                        action = "store"
                        )
    
    parser.add_option("-r", "--root-path",
                        dest = "root_path",
                        help = "Add root path",
                        type = "string",
                        action = "store"
                        )


    options, args = parser.parse_args()
    
    def bailout():
        parser.print_help()
        raise SystemExit

    if not options.input_folder or not options.filename or  not options.start_index or not options.end_index:
        bailout()
    
    input_folder = options.input_folder
    filename = options.filename
    
    if options.output_folder:
        output_folder = options.output_folder
    else:
        output_folder = options.input_folder
        
    start_index = options.start_index
    end_index = options.end_index
    
    filepath = os.path.join(input_folder, filename) 
    cap_temp = cv2.VideoCapture(filepath)
#     fourcc = int(cap_temp.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # For mp4 support while using VideoWriter from opencv
    fps = int(cap_temp.get(cv2.CAP_PROP_FPS))
    size = ( int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT)) )
    print(fourcc, fps, size)
    del(cap_temp)
    
    # Reading video from the start point mentioned
    video_reader = VideoExtractor(input_folder, filename, start_index, end_index)
    
    output_filename = os.path.join(output_folder, filename.split('.')[0] + str(start_index) + '_' +  str(end_index) + '.mp4' )
    print(output_filename)
   
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, size)
    
    success, frame = next(video_reader)
    while success:
        video_writer.write(frame)
        success, frame = next(video_reader)
    
    video_writer.release()
    
    ###############################################################
    #Initialising some paths for next steps
    if options.root_path:
        ROOT_PATH = options.root_path
    FILE_NAME = 'concated.mp4'
    DATASET_PATH = input_folder
    
    
    keypoint_path =  os.path.join(DATASET_PATH, 'concated-keypoints.ndjson')
    tracker_path =  os.path.join(DATASET_PATH, 'concated.ndjson')
    
    tracking_data = ndjson.load(open(tracker_path,'r'))
    keypoint_iterator = iter(ndjson.load(open(keypoint_path,'r')))
    frame_wise_tracker = FramewiseTrackerDictmaker(tracking_data)
    
    
if __name__ == '__main__':
    main()