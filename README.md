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

Make sure to make separate output folder for each dataset to ensure that videos don't get overwrite.

# TODO: 
1. Optimising human estimator. With some small changes and modification, speed and memory usage can be decreased.

