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

    if not options.data_folder or options.output_folder or options.start_index or options.end_index:
        bailout()
    
    
    