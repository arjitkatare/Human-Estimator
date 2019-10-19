# encoding = UTF-8

import cv2
from optparse import OptionParser

def __main__:
    
    parser = OptionParser()
    
    
    parser.add_option("-d", "--data-folder",
                        dest = "data_folder",
                        help = "Add datafolder path",
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
    parser.add_option("-S", "--split-filesize",
                        dest = "split_filesize",
                        help = "Split or chunk size in bytes (approximate)",
                        type = "int",
                        action = "store"
                        )
    parser.add_option("--filesize-factor",
                        dest = "filesize_factor",
                        help = "with --split-filesize, use this factor in time to" \
                               " size heuristics [default: %default]",
                        type = "float",
                        action = "store",
                        default = 0.95
                        )