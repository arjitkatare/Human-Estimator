# Human-Estimator
Computer Vision model to calculate probability of human figure be a real human from openpose keypoints

## How to run

To run the code.

1. Install all the requirements in a virtualenv -

```
pip install -r requirements.txt
```

2. Use following commands from the root path of repo -
```
python main.py -d <input_dir> -f <filename> -o <output_dir> -s <start_index> -e <end_index> -r <root_path> 
```
for my case - this looked like -

```
python main.py -d ../data/mf01d/ -f concated.mp4 -o ./videos/ -s 1000 -e 100000 -r ./
```

# Features used ->
There are 4 custom-made features used in this repo.
1.) Position change feature
2.) Mean keypoints change features
3.) inter keypoints distance features -> this checks for motion in stationary people to check for life(motion)
4.) Zero mask similiarity feature

All other features except zero mask similiarity is understandable by their names. 
## Zero mask similiarity features
This mask majorly checks for continuity of keypoints to ensure that we are only checking changes from same number of keypoints. If this is way to variable and statistical mode of mask is skewed way towards 0 then we can assert that this as motion and defined that id as human

## HumanEstimator.human_certainity (dict)
Once a person is added to human_certainity dictionary then we are sure that body is human and probability is fixed to 1.
We are looking for human_certainity cause this helps us in ensuring that we are tracking a body rather than an inanimated object, Rest of the processing in pipelined for human analysis can simply check for that id in *human_certainity*  and this can be easily done in parallel.

# Notes
1. For the sample data I have adjusted thresholds for 2 of the features ,i.e, mean_keypoints change feature and inter keypoints distance features as primary and secondary decider respectively.
2. A person ID get in the people_getting_tracked dict only after there are some minimum number of frames, here 8. This is done to filter out some random id points being present in the video and this ensure that they are not getting tracked



Make sure to make separate output folder for each dataset to ensure that videos don't get overwrite.

# TODO: 
1. Optimising human estimator. With some small changes and modification, speed and memory usage can be decreased.

