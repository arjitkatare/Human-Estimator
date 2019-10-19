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






def main(DATASET_PATH, keypoint_iterator):
    videopath = os.path.join('/home/user/Workspace/',DATASET_PATH,'concated.mp4')
    basename, ext = os.path.splitext(os.path.basename(videopath))
    out_path =  os.path.join( '/home/user/Workspace/' ,DATASET_PATH , "frames_" + basename)
    os.makedirs(out_path, exist_ok=True)
    vidcap = cv2.VideoCapture(videopath)
    
    #Initialisation before loop
    keypoints_frame_data = next(keypoint_iterator)
    success, image = vidcap.read()
    people = keypoints_frame_data['people']
    
    
    count = 0
    success = True
    while success:
        if success:
            if len(people) != 0:
                image = keypoint_printer(image, people)
            
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
    
    DATASET_PATH = sys.argv[1]
    keypoint_path =  os.path.join( '/home/user/Workspace/', DATASET_PATH, 'concated-keypoints.ndjson')
    print(keypoint_path)
    keypoint_iterator = iter(ndjson.load(open(keypoint_path,'r')))
    main( DATASET_PATH , keypoint_iterator)