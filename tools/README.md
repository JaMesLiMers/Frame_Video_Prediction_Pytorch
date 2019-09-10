# Tools
This folder include the training/testing/demo files, you may want to use these code with the config when training/testing your models. 

# Train

The first thing you need to do is build the layers and abstract class in model folder, implement your model in experiments/'experiment_name'/Custom.py file, and the model class name should be the 'Custom'.

After build your own Custom class, you can specify your experiment folder name ('expriment_name') in config file. Then run the train_video_prediction.py file.

if you want to customize your class name, you can change the training file code to adjust your model arch.

A config file template is also included in this folder.
