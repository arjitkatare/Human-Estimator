# encoding = UTF-8

import os
import sys
import cv2
from optparse import OptionParser

def VideoExtractor(input_folder, filename, start_index, end_index):
    filepath = os.path.join(input_folder, filename)
    
    #Initialisation
    cap = cv2.VideoCapture(filepath)
    frame_number = start_index
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index + 1)
    
    while frame_number != end_index: 

        yield cap.read()
        frame_number+=1
    
    yield (False, None)

    
    
    