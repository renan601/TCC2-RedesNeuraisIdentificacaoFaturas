# Neural network to identify blocks of information in invoices (TCC2)
This project uses a RetinaNet + Resnet-50 to identify the address, consumer unit number, billing table, taxes table and readings table in invoices from two brazilian distributors (CPFL and CEMIG).

The private dataset used for the training and validation process had 10 thousand PDFs from the distributors. The PDFs were transformed into 640x640x3 PNG images for the training process.

The network loads a JSON containg a path to image files and bounding box identification for each image. Bounding box uses a format relXmin, relYmin, relXmax, relYmax, where each coordinate is divided by 640 (image size).

Use ```retinaNet/train.py``` to train the network with personalized data.

Use ```evaluate.py``` to get metrics like AP, mAP, recall and precision from the validation dataset.

Use ```predict.py``` to predict bbox and classes for a new image.

Two networks were trained to better adapt to input data. One for low voltage invoices (LV) and other for medium/high (MV) voltage invoices.

The AP and mAP for LV invoices:
* AP Endereço: 0.95
* AP Instalação: 0.85
* AP Faturamento: 0.88798
* AP Impostos: 0.9
* AP Leitura: 0.9
* mAP: 0.8975

The AP and mAP for MV invoices:
* AP Endereço: 0.7332
* AP Instalação: 0.7737
* AP Faturamento: 0.61056
* AP Impostos: 0.9849
* AP Leitura: 0.5184
* mAP: 0.72415

# How to Install Drivers / CUDA / CUDNN on Nvidia GPU to run TensorFlow

### To verify your gpu is cuda enable check
```lspci | grep -i nvidia```

### If you have previous installation remove it first. 
```sudo apt purge nvidia* -y```

```sudo apt remove nvidia-* -y```

```sudo rm /etc/apt/sources.list.d/cuda*```

```sudo apt autoremove -y && sudo apt autoclean -y```

```sudo rm -rf /usr/local/cuda*```

### System update
```sudo apt update && sudo apt upgrade -y```

### Install other import packages
```sudo apt install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev```

### Get the PPA repository driver
```sudo add-apt-repository ppa:graphics-drivers/ppa```

```sudo apt update```

### Find recommended driver versions for you
```ubuntu-drivers devices```

### Install nvidia driver with dependencies
```sudo apt install libnvidia-common-515 libnvidia-gl-515 nvidia-driver-515 -y```

### Reboot
```sudo reboot now```

### Verify that the following command works
```nvidia-smi```

```sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin```

```sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600```

```sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub```

```sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"```

### Update and upgrade
```sudo apt update && sudo apt upgrade -y```

### Installing CUDA-11.8
```sudo apt install cuda-11-8 -y```

### Setup your paths
```echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc```

```echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc```

```source ~/.bashrc```

```sudo ldconfig```

### Install cuDNN v11.8
### First register here: https://developer.nvidia.com/developer-program/signup

```CUDNN_TAR_FILE="cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz"```

```sudo wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz```

```sudo tar -xvf ${CUDNN_TAR_FILE}```

```sudo mv cudnn-linux-x86_64-8.7.0.84_cuda11-archive cuda```

### Copy the following files into the cuda toolkit directory.
```sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.8/include```

```sudo cp -P cuda/lib/libcudnn* /usr/local/cuda-11.8/lib64/```

```sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*```

### Finally, to verify the installation, check
```nvidia-smi```

```nvcc -V```

# Install Packages

### Create a virtual env in Python (3.10)

```python3 -m venv envName```

### Install all depedencies

```pip install -r /path/to/requirements.txt```