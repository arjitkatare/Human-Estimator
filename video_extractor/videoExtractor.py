# encoding = UTF-8

import os
import sys
import cv2
from optparse import OptionParser

def VideoExtractor(data_folder, filename, start_index, end_index):
    filepath = os.path.join(data_folder, filename)
    
    #Initialisation
    cap = cv2.VideoCapture(filepath)
    frame_number = start_index
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    
    while frame_number != end_index+1: 

        yield cap.read()
    
    yield (False, None)

    
    
    