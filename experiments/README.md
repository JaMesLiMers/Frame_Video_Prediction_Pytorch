# Experiments
This folder is to store the experiments models

You should specify the 'experiment_name' tag as your own folder in the config file when you're using your own models to train.

If you didn't specify the 'arch' tag in the config file, the training program will automatically find the Custom class in $ROOT/experiments/'experiment_name'/Custom.py as the implemented models.

If you want to use other arch, you may want to change the training code for your own arch.
