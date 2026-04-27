# FDSNet
This is official Pytorch implementation of "FDSNet: Frequency-Domain Decomposition and State Space Modeling for Infrared and Visible Image Fusion"

# Framework
The overall framework of the proposed FDSNet.



## Recommended Environment
Environment required for installing code

Run `pip install -r requirements.txt`

 - [ ] torch>=1.13.0
 - [ ] torchvision>=0.14.0
 - [ ] numpy>=1.21.0
 - [ ] Pillow>=9.0.0
 - [ ] tqdm>=4.64.0
 - [ ] einops>=0.6.0
 - [ ] PyWavelets>=1.4.0
 - [ ] natsort>=8.2.0
 - [ ] mamba-ssm>=1.2.0


# To Train
Modify training, testing parameters, and dataset paths within `options.py`

Run `python train.py`.

# To Test

Run `python test.py`.
