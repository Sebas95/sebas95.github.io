
# EffConvNet

## About
High efficiency convolutional neural network model for image recognition in CIFAR-10


## Usage
To train the model from scratch don't specify a file location

'''
python main.py --epochs 10
'''

To load model from existing file. Usage example:
'''
python main.py --model-path './checkpoints/persistedmodel.pth'
'''


# How to create the environment
conda env create -f environment.yml

#### Enter the environment
conda activate mlprojects

#### Exit the environment
conda deactivate

#### Delete the environment
conda remove --name mlprojects --all
