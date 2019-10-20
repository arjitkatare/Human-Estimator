# encoding = UTF-8
import os
import sys
import cv2
import ndjson





def FramewiseIdPrinter(image, frame_wise_tracker, count):
    total_positions = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0) 
    fontScale = 1
    thickness = 2
    if str(count) in frame_wise_tracker:
        people_positions = frame_wise_tracker[str(count)]
    
        for people in people_positions:
            position = people['position']
            pId = people['id']
            
            image = cv2.putText(image, str(pId), (int(position[0]), int(position[1])), font, fontScale, color, thickness, cv2.LINE_AA   )
    return image


def FramewiseTrackerDictmaker(a):
    frame_wise_tracker = {} 

    def fwt_dictmaker(peop_id, frame_pose):
        return {
            'id': peop_id,
            'pose': frame_pose,
            'position': frame_position
        }


    for people in a:
        peop_id = people['id']
        peop_frames = people['frames']

        for frame in peop_frames:
            frame_id = frame['timestamp']
            frame_pose = frame['pose']
            frame_position = frame['position']

            temp_dict = fwt_dictmaker(peop_id, frame_pose)

            if frame_id in frame_wise_tracker:
                frame_wise_tracker[frame_id].append(temp_dict)
            else:
                frame_wise_tracker[frame_id] = [temp_dict]
    
    return frame_wise_tracker