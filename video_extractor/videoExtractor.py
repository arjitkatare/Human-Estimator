# encoding = UTF-8

import os
import sys
import cv2
from optparse import OptionParser

def VideoExtractor(data_folder, filename, output_folder, start_index, end_index):
    filepath = os.path.join(data_folder, filename)
    
    #Initialisation
    cap = cv2.VideoCapture(fielpath)
    frame_number = start_index
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    
    while frame_number != end_index+1: 
        success, image = cap.read()
        
        if success:
            yield


def __main__:
    
    parser = OptionParser()
    
    
    parser.add_option("-d", "--data-folder",
                        dest = "data_folder",
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
    options, args = parser.parse_args()
    
    def bailout():
        parser.print_help()
        raise SystemExit

    if not options.data_folder or not options.filename or  not options.start_index or not options.end_index:
        bailout()
    
    if not options.output_folder:
        VideoExtractor(options.input_folder, options.filename, options.input_folder, options.start_index, options.end_index)
    else:
        VideoExtractor(options.input_folder, options.filename, options.output_folder, options.start_index, options.end_index)
    
    
    