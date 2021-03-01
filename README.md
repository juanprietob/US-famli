# US-famli
Ultra sound Fetal Age Machine Learning Initiative

## Installation

1. Install docker with tensorflow and gpu support

2. Start a container as root (without the flag -u $(id -u):$(id -g))

3. Install other required packages: 

3.1 pip install itk vtk pandas sklearn matplotlib

3.2 Install x11 and gl libraries: apt-get update; apt-get install libx11-6 libgl1 libopengl0

3.3 If installing vtk egl the path to python packages is at /usr/local/lib/python3.6/dist-packages

4. Commit the container and use this one, remember to start it as normal user (use flag -u $(id -u):$(id -g)). 

5. Installing with conda

5.1 conda install vtk pandas scikit-learn matplotlib

5.2 conda install tensorflow==2.2

5.2 conda install -c conda-forge itk


## How to train a network from scratch

1. Create CSV file with columns img,class or img,seg

2. Convert to tfRecords US-famli/src/py/dl/tfRecords.py

3. Train a network with US-famli/src/py/dl/train_v2.py

