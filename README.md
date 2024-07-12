
# HoVer-Net as an histo-miner submodule

Here is the presentation of Hovernet model, as a submodule of histo-miner. The original code is coming from: [hover-ner repository](https://github.com/vqdang/hover_net).
This fork contains some variation corresponding to the specific training used for histo-miner pipeline. 


## Presentation of Hovernet

A multiple branch network that performs nuclear instance segmentation and classification within a single network. The network leverages the horizontal and vertical distances of nuclear pixels to their centres of mass to separate clustered cells. A dedicated up-sampling branch is used to classify the nuclear type for each segmented instance. <br />


## Installation from yml file

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


## Installation from scratch

Alternatively the hover-net env can be installed from scratch as follow:

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




