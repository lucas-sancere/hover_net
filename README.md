
# HoVer-Net as an histo-miner submodule

Here is the presentation of Hovernet model, as a submodule of histo-miner. The original code is coming from: [enter].
This fork contains some variation corresponding to the specific training used for histo-miner pipeline. 


## Presentation of Hovernet

This README is a copy of the original readme one can find [enter link].

A multiple branch network that performs nuclear instance segmentation and classification within a single network. The network leverages the horizontal and vertical distances of nuclear pixels to their centres of mass to separate clustered cells. A dedicated up-sampling branch is used to classify the nuclear type for each segmented instance. <br />


## Installation new

One can install the required dependencies inside a conda env as follow:

**Step 1.**Open a terminal in your local or remote device, create hovernet_submodule environment using the corresponding yml congif and activate it: 
```shell
conda env create -f hovernet_submodule.yml
conda activate hovernet_submodule
```

**Step 2.** Install PyTorch ([official instructions](https://pytorch.org/get-started/locally/)):

- **Recommended:** install PyTorch version 1.10 with CUDA 11.3 (tested on V100 and A100 GPUs):
```shell
pip install torch==1.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

- **Not Recommended: **Original env from hovernet repo [enter], install  PyTorch version 1.6 with CUDA 10.2 (not working on A100 and younger):
```shell
pip install torch==1.6.0 torchvision==0.7.0
```



## Installation old

One can install the required dependencies inside a conda env as follow:

**Step 1.**Open a terminal in your local or remote device, create a conda environment and activate it. 

```shell
conda create -n hovernet_submodule python==3.6.12 -y
conda activate hovernet_submodule
```

**Step 2.** Install PyTorch ([official instructions](https://pytorch.org/get-started/locally/)):

- **Recommended:** install PyTorch version 1.10 with CUDA 11.3 (tested on V100 and A100 GPUs):
```shell
pip install torch==1.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

- **Not Recommended: **Original env from hovernet repo [enter], install  PyTorch version 1.6 with CUDA 10.2 (not working on A100 and younger):
```shell
pip install torch==1.6.0 torchvision==0.7.0
```

**Step 3**. Install conda dependancies
```shell
conda install openslide==3.4.1
conda install libgcc_mutex==0.1
conda install _openmp_mutex=4.5
conda install ca-certificates==2023.7.22
conda install cairo==1.16.0 
conda install expat==2.5.0  
conda install fontconfig==2.14.2
conda install freetype==2.12.1
conda install gdk-pixbuf==2.42.6 
conda install gettext==0.21.1 
conda install glib==2.68.4
conda install glib-tools==2.68.4
conda install icu==68.2 
conda install jpeg==9e
conda install ld_impl_linux-64==2.40
conda install lerc==4.0.0
conda install libdeflate==1.14
conda install libexpat==2.5.0
conda install libgcc-ng==13.1.0
conda install libglib==2.68.4 
conda install libgomp==13.1.0
conda install libiconv==1.17     
conda install libpng==1.6.39
conda install libsqlite==3.42.0
conda install libstdcxx-ng==13.1.0
conda install libtiff==4.4.0
conda install libuuid==2.38.1
conda install libwebp-base==1.3.1
conda install libxcb==1.15 
conda install libxml2==2.9.12
conda install libzlib==1.2.13
conda install ncurses==6.4
conda install openjpeg==2.5.0                                                     
conda install openslide==3.4.1                                                                                         
conda install openssl==1.1.1v
conda install pcre==8.45
conda install pip==21.3.1
conda install pixman==0.40.0
conda install pthread-stubs==0.4
conda install python==3.6.12
conda install python_abi==3.6
conda install readline==8.2                       
conda install setuptools==58.0.4
conda install sqlite==3.42.0
conda install tk==8.6.12
conda install wheel==0.37.1
conda install xorg-kbproto==1.0.7
conda install xorg-libice==1.1.1
conda install xorg-libsm==1.2.4
conda install xorg-libx11==1.8.6
conda install xorg-libxau==1.0.11
conda install xorg-libxdmcp==1.1.3
conda install xorg-libxext==1.3.4
conda install xorg-libxrender==0.9.11
conda install xorg-renderproto==0.11.1
conda install xorg-xextproto==7.3.0
conda install xorg-xproto==7.0.31
conda install xz==5.2.6
conda install zlib==1.2.13
conda install zstd==1.5.5

```

**Step 4**. Install all dependancies. Still inside the `hovernet_submodule` env run:

```shell
pip install -r requirements.txt
```


## Note for pytorch version for GPUs H100 and more recent









--------- Original README ---

-----------------------------------------------------------------------------


[Link](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub) to Medical Image Analysis paper. <br />

This is the official PyTorch implementation of HoVer-Net. For the original TensorFlow version of this code, please refer to [this branch](https://github.com/vqdang/hover_net/tree/tensorflow-final). The repository can be used for training HoVer-Net and to process image tiles or whole-slide images. As part of this repository, we supply model weights trained on the following datasets:

- [CoNSeP](https://www.sciencedirect.com/science/article/pii/S1361841519301045)
- [PanNuke](https://arxiv.org/abs/2003.10778)
- [MoNuSAC](https://ieeexplore.ieee.org/abstract/document/8880654)
- [Kumar](https://ieeexplore.ieee.org/abstract/document/7872382)
- [CPM17](https://www.frontiersin.org/articles/10.3389/fbioe.2019.00053/full)

Links to the checkpoints can be found in the inference description below.

![](docs/diagram.png)

## Set Up Environment

```
conda env create -f environment.yml
conda activate hovernet
pip install torch==1.6.0 torchvision==0.7.0
```

Above, we install PyTorch version 1.6 with CUDA 10.2. 

## Repository Structure

Below are the main directories in the repository: 

- `dataloader/`: the data loader and augmentation pipeline
- `docs/`: figures/GIFs used in the repo
- `metrics/`: scripts for metric calculation
- `misc/`: utils that are
- `models/`: model definition, along with the main run step and hyperparameter settings  
- `run_utils/`: defines the train/validation loop and callbacks 

Below are the main executable scripts in the repository:

- `config.py`: configuration file
- `dataset.py`: defines the dataset classes 
- `extract_patches.py`: extracts patches from original images
- `compute_stats.py`: main metric computation script
- `run_train.py`: main training script
- `run_infer.py`: main inference script for tile and WSI processing
- `convert_chkpt_tf2pytorch`: convert tensorflow `.npz` model trained in original repository to pytorch supported `.tar` format.

# Running the Code

## Training

### Data Format
For training, patches must be extracted using `extract_patches.py`. For instance segmentation, patches are stored as a 4 dimensional numpy array with channels [RGB, inst]. Here, inst is the instance segmentation ground truth. I.e pixels range from 0 to N, where 0 is background and N is the number of nuclear instances for that particular image. 

For simultaneous instance segmentation and classification, patches are stored as a 5 dimensional numpy array with channels [RGB, inst, type]. Here, type is the ground truth of the nuclear type. I.e every pixel ranges from 0-K, where 0 is background and K is the number of classes.

Before training:

- Set path to the data directories in `config.py`
- Set path where checkpoints will be saved  in `config.py`
- Set path to pretrained Preact-ResNet50 weights in `models/hovernet/opt.py`. Download the weights [here](https://drive.google.com/file/d/1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5/view?usp=sharing).
- Modify hyperparameters, including number of epochs and learning rate in `models/hovernet/opt.py`.

### Usage and Options

Usage: <br />
```
  python run_train.py [--gpu=<id>] [--view=<dset>]
  python run_train.py (-h | --help)
  python run_train.py --version
```

Options:
```
  -h --help       Show this string.
  --version       Show version.
  --gpu=<id>      Comma separated GPU list.  
  --view=<dset>   Visualise images after augmentation. Choose 'train' or 'valid'.
```

Examples:

To visualise the training dataset as a sanity check before training use:
```
python run_train.py --view='train'
```

To initialise the training script with GPUs 0 and 1, the command is:
```
python run_train.py --gpu='0,1' 
```

## Inference

### Data Format
Input: <br />
- Standard images files, including `png`, `jpg` and `tiff`.
- WSIs supported by [OpenSlide](https://openslide.org/), including `svs`, `tif`, `ndpi` and `mrxs`.

Output: <br />
- Both image tiles and whole-slide images output a `json` file with keys:
    - 'bbox': bounding box coordinates for each nucleus
    - 'centroid': centroid coordinates for each nucleus
    - 'contour': contour coordinates for each nucleus 
    - 'type_prob': per class probabilities for each nucleus (default configuration doesn't output this)
    - 'type': prediction of category for each nucleus
- Image tiles output a `mat` file, with keys:
    - 'raw': raw output of network (default configuration doesn't output this)
    - 'inst_map': instance map containing values from 0 to N, where N is the number of nuclei
    - 'inst_type': list of length N containing predictions for each nucleus
 - Image tiles output a `png` overlay of nuclear boundaries on top of original RGB image

### Model Weights

Model weights obtained from training HoVer-Net as a result of the above instructions can be supplied to process input images / WSIs. Alternatively, any of the below pre-trained model weights can be used to process the data. These checkpoints were initially trained using TensorFlow and were converted using `convert_chkpt_tf2pytorch.py`. Provided checkpoints either are either trained for segmentation alone or for simultaneous segmentation and classification. Note, we do not provide a segmentation and classification model for CPM17 and Kumar because classification labels aren't available.

**IMPORTANT:** CoNSeP, Kumar and CPM17 checkpoints use the original model mode, whereas PanNuke and MoNuSAC use the fast model mode. Refer to the inference instructions below for more information. 

Segmentation and Classification:
- [CoNSeP checkpoint](https://drive.google.com/file/d/1FtoTDDnuZShZmQujjaFSLVJLD5sAh2_P/view?usp=sharing)
- [PanNuke checkpoint](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view?usp=sharing)
- [MoNuSAC checkpoint](https://drive.google.com/file/d/13qkxDqv7CUqxN-l5CpeFVmc24mDw6CeV/view?usp=sharing)

Segmentation Only:
- [CoNSeP checkpoint](https://drive.google.com/file/d/1BF0GIgNGYpfyqEyU0jMsA6MqcUpVQx0b/view?usp=sharing)
- [Kumar checkpoint](https://drive.google.com/file/d/1NUnO4oQRGL-b0fyzlT8LKZzo6KJD0_6X/view?usp=sharing) 
- [CPM17 checkpoint](https://drive.google.com/file/d/1lR7yJbEwnF6qP8zu4lrmRPukylw9g-Ms/view?usp=sharing) 

Access the entire checkpoint directory, along with a README on the filename description [here](https://drive.google.com/drive/folders/17IBOqdImvZ7Phe0ZdC5U1vwPFJFkttWp?usp=sharing).

If any of the above checkpoints are used, please ensure to cite the corresponding paper.

### Usage and Options

Usage: <br />
```
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)
```

Options:
```
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlay color. [default: '']

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used in PanNuke / MoNuSAC, 'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size. [default: 128]
```

Tile Processing Options: <br />
```
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --draw_dot             To draw nuclei centroid on overlay. [default: False]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: False]
   --save_raw_map         To save raw prediction or not. [default: False]
```

WSI Processing Options: <br />
```
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 10000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 2048]
    --save_thumb            To save thumb. [default: False]
    --save_mask             To save mask. [default: False]
```

The above command can be used from the command line or via an executable script. We supply two example executable scripts: one for tile processing and one for WSI processing. To run the scripts, first make them executable by using `chmod +x run_tile.sh` and `chmod +x run_tile.sh`. Then run by using `./run_tile.sh` and `./run_wsi.sh`.

Intermediate results are stored in cache. Therefore ensure that the specified cache location has enough space! Preferably ensure that the cache location is SSD.

Note, it is important to select the correct model mode when running inference. 'original' model mode refers to the method described in the original medical image analysis paper with a 270x270 patch input and 80x80 patch output. 'fast' model mode uses a 256x256 patch input and 164x164 patch output. Model checkpoints trained on Kumar, CPM17 and CoNSeP are from our original publication and therefore the 'original' mode **must** be used. For PanNuke and MoNuSAC, the 'fast' mode **must** be selected. The model mode for each checkpoint that we provide is given in the filename. Also, if using a model trained only for segmentation, `nr_types` must be set to 0.

`type_info.json` is used to specify what RGB colours are used in the overlay. Make sure to modify this for different datasets and if you would like to generally control overlay boundary colours.

As part of our tile processing implementation, we add an option to save the output in a form compatible with QuPath. 

Take a look on how to utilise the output in `examples/usage.ipynb`. 

## Overlaid Segmentation and Classification Prediction

<p float="left">
  <img src="docs/seg.gif" alt="Segmentation" width="870" />
</p>

Overlaid results of HoVer-Net trained on the CoNSeP dataset. The colour of the nuclear boundary denotes the type of nucleus. <br />
Blue: epithelial<br />
Red: inflammatory <br />
Green: spindle-shaped <br />
Cyan: miscellaneous

## Datasets

Download the CoNSeP dataset as used in our paper from [this link](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/). <br />
Download the Kumar, CPM-15, CPM-17 and TNBC datsets from [this link](https://drive.google.com/open?id=1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK).  <br />
Down

Ground truth files are in `.mat` format, refer to the README included with the datasets for further information. 

## Comparison to Original TensorFlow Implementation

Below we report the difference in segmentation results trained using this repository (PyTorch) and the results reported in the original manuscript (TensorFlow). 

Segmentation results on the Kumar dataset:
| Platform   | DICE       | PQ         | AJI       |
| -----------|----------- | -----------|-----------|
| TensorFlow | 0.8258     | 0.5971     | 0.6412    |
| PyTorch    | 0.8211     | 0.5904     | 0.6321    |

Segmentation results on CoNSeP dataset: 
| Platform   | DICE       | PQ         | AJI       |
| -----------|----------- | -----------|-----------|
| TensorFlow | 0.8525     | 0.5477     | 0.5995    |
| PyTorch    | 0.8504     | 0.5464     | 0.6009    |

Checkpoints to reproduce the above results can be found [here](https://drive.google.com/drive/folders/17IBOqdImvZ7Phe0ZdC5U1vwPFJFkttWp?usp=sharing).

Simultaneous Segmentation and Classification results on CoNSeP dataset: 
| Platform   | F1<sub>d</sub> | F1<sub>e</sub> | F1<sub>i</sub> | F1<sub>s</sub> | F1<sub>m</sub> |
| -----------|----------------| ---------------|----------------|----------------|----------------|
| TensorFlow | 0.748          | 0.635          | 0.631          | 0.566          | 0.426          |
| PyTorch    | 0.756          | 0.636          | 0.559          | 0.557          | 0.348          |


## Citation

If any part of this code is used, please give appropriate citation to our paper. <br />

BibTex entry: <br />
```
@article{graham2019hover,
  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak, Jin Tae and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  pages={101563},
  year={2019},
  publisher={Elsevier}
}
```

## Authors

* [Quoc Dang Vu](https://github.com/vqdang)
* [Simon Graham](https://github.com/simongraham)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

Note that the PanNuke dataset is licensed under [Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/), therefore the derived weights for HoVer-Net are also shared under the same license. Please consider the implications of using the weights under this license on your work and it's licensing. 


--------------------------------------------------------------------------------
