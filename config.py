import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self, optname='opt'):
        self.seed = 10
        self.optname = optname

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        model_name = "hovernet"
        model_mode = "fast" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = 6 # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [164, 164] # patch shape at output of network

        if model_mode == "original":
            if act_shape != [270, 270] or out_shape != [80, 80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256, 256] or out_shape != [164, 164]:
                raise Exception("If using `original` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.dataset_name = "dlbcl_mc_chrisfix" # extracts dataset info from dataset.py
        self.log_dir = "logs_DLBCLChris/{}/".format(self.optname) # where checkpoints will be saved
        # self.dataset_name = "dlbcl_mc1" # extracts dataset info from dataset.py
        # self.log_dir = "logs_DLBCLMC1/{}/".format(self.optname) # where checkpoints will be saved

        # paths to training and validation patches
        self.train_dir_list = [
            "/data/lsancere/Hover_Net_Complete/pytorch-final/hover_net/dataset/training_data/dlbcl_mc_chrisfix/dlbcl_mc_chrisfix/train/540x540_164x164/"
        ]
        self.valid_dir_list = [
            "/data/lsancere/Hover_Net_Complete/pytorch-final/hover_net/dataset/training_data/dlbcl_mc_chrisfix/dlbcl_mc_chrisfix/valid/540x540_164x164/"
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape, },
            "valid": {"input_shape": act_shape, "mask_shape": out_shape, },
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.{}.{}".format(model_name, self.optname)
            )
        self.model_config = module.get_config(nr_type, model_mode)
