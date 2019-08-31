import json
import logging
from os.path import exists


class Configs(object):
    def __init__(self, file_path_or_dict, logger_name="global"):
        super(Configs, self).__init__()
        # 加载logger
        self.logger = logging.getLogger(logger_name)
        # 加载cfg文件, 处理基础的参数
        self.check_meta(self.load_config(file_path_or_dict))

    def load_config(self, file_path_or_dict):
        if type(file_path_or_dict) is str:
            assert exists(file_path_or_dict), '"{}" not exists'.format(file_path_or_dict) # 确认有文件
            config = dict(json.load(open(file_path_or_dict))) # 打开加载json文件
        elif type(file_path_or_dict) is dict:
            config = file_path_or_dict
        else:
            raise Exception("The input must be a string path or a dict")
        return config

    def check_meta(self, cfg_init):
        """
        Check 是否将基础的需求写入了config文件中, 没有的话就使用默认的.
        具体check的有:
        是否包含meta标签, 以及重要的experiment_path和arch.
        如果没问题的话就赋值给Config类.
        """
        self.cfg_init = cfg_init

        # check meta部分
        if 'meta' not in self.cfg_init:
            self.logger.warning("The cfg not include meta tag, will generate default")
            self.logger.warning("Used the default meta configs.")
            self.cfg_init['meta'] = {"experiment_path": "experiments",
                                "arch": "Custom",
                                "board_path": "board",
                                "experiment_name":"video_prediction_demo"},
        else:
            cfg_meta = self.cfg_init['meta']
            if 'experiment_path' not in cfg_meta:
                self.logger.warning("Not specified experiment_path, used default. ('experiments')")
                self.cfg_init['meta']['experiment_path'] = "experiments"
            if 'arch' not in cfg_meta:
                self.logger.warning("Not specified arch, used default. (Custom)")
                self.cfg_init['meta']['arch'] = "Custom"
            if 'board_path' not in cfg_meta:
                self.logger.warning("Not specified board_path, used default. (board)")
                self.cfg_init['meta']['board_path'] = "board"
            if 'experiment_name' not in cfg_meta:
                self.logger.warning("Not specified experiment_name, used default. (video_prediction_demo)")
                self.cfg_init['meta']['experiment_name'] = "video_prediction_demo"

        # 赋值部分
        self.__dict__.update(self.cfg_init)

    def log_dict(self, logger_name="global"):
        # 打印Config的文件到log
        self.logger.debug("Used config: \n {}".format(self.cfg_init))



if __name__ == "__main__":
    """
    cfg 文件的加载类
    cfg文件的格式为json文件或者dict()

    meta部分标明实验的元信息(必须, 会验证),
    train部分标明训练时候的参数,
    model部分标明模型的参数

    下面是例子和使用方法:
    """

    test_cfg = {
                "meta":{
                    "experiment_path": "./experiments/video_prediction/",
                    "arch": "Custom"
                },
                "train":{
                    "epoches": 50,
                    "batch_size": 16,
                    "lr": 1e-3
                },
                "model":{
                    "input_size":[64,64],
                    "input_dim":1,
                    "hidden_dim":64,
                    "kernel_size":[3,3],
                    "num_layers":3,
                    "bias":True,
                    "return_all_layers": False,
                    "predict_num": 10
                }
            }
    test_Configs = Configs(test_cfg)

    # 直接访问名字即可, 返回的是一个dict
    print(test_Configs.meta)
    print(type(test_Configs.meta))

    print(test_Configs.train)    
    print(type(test_Configs.train))

    print(test_Configs.model)    
    print(type(test_Configs.model))