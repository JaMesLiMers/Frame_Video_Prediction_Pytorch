# Frame_Video_Prediction_Pytorch
frame video prediction algorithm framework used pytorch.
This project is supposed to be a example of dl to help my frined finish their surf project.
Hope this can help you too~~

# How to start
Walkthrough the folders ReadMe.md file for a quick look.

for the first time, you may want to build up a conda environment use:

`conda create -n torchenv python=3.6`

`source activate torchenv`

`pip install -r requirements.txt`

Then run our demo derectly:

`python3 ./tools/train_video_prediction.py`

The data will be download to ./data and the result will be stored in ./board using tensorboardX and ./logs.

You can use:

`tensorboard --logdir ./board/video_prediction_demo/`

to see the results.

PS: we assume our running root path is current level.
