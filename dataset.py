import glob
import cv2
import numpy as np
import scipy.io as sio


class __AbstractDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.
    
    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError


####
class __Kumar(__AbstractDataset):
    """Defines the Kumar dataset as originally introduced in:

    Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane, 
    and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for 
    computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560.

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class __CPM17(__AbstractDataset):
    """Defines the CPM 2017 dataset as originally introduced in:

    Vu, Quoc Dang, Simon Graham, Tahsin Kurc, Minh Nguyen Nhat To, Muhammad Shaban, 
    Talha Qaiser, Navid Alemi Koohbanani et al. "Methods for segmentation and classification 
    of digital microscopy tissue images." Frontiers in bioengineering and biotechnology 7 (2019).

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class __CoNSeP(__AbstractDataset):
    """Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in 
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563
    
    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        if with_type:
            ann_type = sio.loadmat(path)["type_map"]

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann

####
class __PANNUKEskin(__AbstractDataset):
    """PANNUKE data keeping only skin images
    
    """

####
class __DLBCL(__AbstractDataset):
    """DLBCL data set
    
    """

####
class __DLBCL3(__AbstractDataset):
    """DLBCL data set
    
    """

####
class __DLBCL4(__AbstractDataset):
    """DLBCL data set
    
    """

####
class __DLBCLMC1(__AbstractDataset):
    """DLBCL data set
    
    """

####
class __DLBCLMCCHRIS(__AbstractDataset):
    """dlbcl_mc_chris data set
    
    Dataset coming from the generation of the annotated data by Chris. 
    """

####
class __DLBCLMCCHRIS_FIX(__AbstractDataset):
    """dlbcl_mc_chrisfix data set
    
    dlbcl_mc_chris where some data were deleted because generated a bug.
    The following data that was deleted is: 2021-05-07_18.03.28_Region2 instance map
    And: 2021-05-07_21.28.14_Region1_Tile32 class map
    """


####
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "kumar": lambda: __Kumar(),
        "cpm17": lambda: __CPM17(),
        "consep": lambda: __CoNSeP(),
        "pannukeskin": lambda: __PANNUKEskin(),
        "dlbcl": lambda: __DLBCL(),
        "dlbcl3": lambda: __DLBCL3(),
        "dlbcl4": lambda: __DLBCL4(),
        "dlbcl_mc1": lambda: __DLBCLMC1(),
        "dlbcl_mc_chris": lambda: __DLBCLMCCHRIS(),
        "dlbcl_mc_chrisfix": lambda: __DLBCLMCCHRIS_FIX(),
        # "dlbcl_mc_chrisfix_noobg": lambda: __DLBCLMCCHRIS_FIX_NOOBG(), 
        # "dlbcl_mc_test1": lambda: __TEST1(),
        # "dlbcl_mc_test11": lambda: __TEST11(),
        # "dlbcl_mc_test12": lambda: __TEST12(),
        # "dlbcl_mc_test13": lambda: __TEST13(),
        # "dlbcl_mc_test14": lambda: __TEST14(),
        # "dlbcl_mc_test15": lambda: __TEST15(),
        # "dlbcl_mc_test16": lambda: __TEST16(),
        # "dlbcl_mc_test17": lambda: __TEST17(),
        # "dlbcl_mc_test2": lambda: __TEST2(),
        # "dlbcl_mc_test3": lambda: __TEST3(),
        # "dlbcl_mc_test4": lambda: __TEST4(),
        # "dlbcl_mc_test5": lambda: __TEST5(),
        # "dlbcl_mc_test6": lambda: __TEST6(),
        # "dlbcl_mc_test7": lambda: __TEST7(),

    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name



# ####
# class __DLBCLMCCHRIS_FIX_NOOBG(__AbstractDataset):
#     """dlbcl_mc_chrisfix_noobg data set
    
#     """

# ####
# class __TEST1(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST11(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST12(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST13(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST14(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST15(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST16(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST17(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST2(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST3(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST4(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST5(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST6(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """

# ####
# class __TEST7(__AbstractDataset):
#     """Dataset to find the corrupted data on __DLBCLMCCHRIS_FIX dataset
    
#     """
