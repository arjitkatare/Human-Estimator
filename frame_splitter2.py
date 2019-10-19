"""Extract frames from video using opencv and plotting keypoints from given data into given video frame and storing it as an image for reference and analysis
"""


import os
import sys
import cv2
import ndjson

def keypoint_printer(image, people):
    
    total_keypoints = []
    chunk_size = 3
    n = chunk_size
    for poses in people:
        keypoints = poses['pose_keypoints']
        keypoints = [keypoints[i * n:(i + 1) * n] for i in range((len(keypoints) + n - 1) // n )] 
        for point in keypoints:
            temp = cv2.KeyPoint(point[0], point[1], _size = 1)
            total_keypoints.append(temp)
    cv2.drawKeypoints(image,total_keypoints, image)
    return image

def framewiseId_printer(image, frame_wise_tracker, count):
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
    


def frameWiseTracker_dictmaker(a):
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


def main(DATASET_PATH, keypoint_iterator, frame_wise_tracker):
    videopath = os.path.join(ROOT_PATH,DATASET_PATH,FILENAME)
    basename, ext = os.path.splitext(os.path.basename(videopath))
    out_path =  os.path.join( ROOT_PATH ,DATASET_PATH , "frames_" + basename)
    os.makedirs(out_path, exist_ok=True)
    vidcap = cv2.VideoCapture(videopath)
    
    #Initialisation before loop
    keypoints_frame_data = next(keypoint_iterator) # Using a generator shall be prefered whenever possible
    success, image = vidcap.read()
    people = keypoints_frame_data['people']
    
    
    count = 0
    success = True
    while success:
        if success:
            if len(people) != 0:
                image = keypoint_printer(image, people)
                image = framewiseId_printer(image, frame_wise_tracker, count)
            
            cv2.imwrite(
                os.path.join(out_path,
                             basename + "_frame{}.jpg".format(count)), image)
        success, image = vidcap.read()
        keypoints_frame_data = next(keypoint_iterator)
        people = keypoints_frame_data['people']
        
        if count % 1000 == 0 :
            print('Read a new frame {}: {}'.format(count, success), end="\r")
        count += 1

if __name__ == "__main__":
    
    try:
        ROOT_PATH = sys.argv[2]
    except:
        ROOT_PATH = '/home/user/Workspace/'
    
    try:
        FILENAME = sys.argv[3]
    except:
        FILENAME = 'concated.mp4'
    
    DATASET_PATH = sys.argv[1]
    keypoint_path =  os.path.join( ROOT_PATH, DATASET_PATH, 'concated-keypoints.ndjson')
    tracker_path =  os.path.join(ROOT_PATH, DATASET_PATH, 'concated.ndjson')
    a = ndjson.load(open(tracker_path,'r'))
    
    print(keypoint_path)
    
    
    keypoint_iterator = iter(ndjson.load(open(keypoint_path,'r')))
    
    frame_wise_tracker = frameWiseTracker_dictmaker(a)
    
    main( DATASET_PATH , keypoint_iterator, frame_wise_tracker)