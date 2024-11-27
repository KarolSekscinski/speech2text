#!/bin/bash


# Install cuDNN
echo "Installing cuDNN..."
CUDNN_VERSION="8.6.0"
CUDNN_TARBALL="cudnn-linux-x86_64-${CUDNN_VERSION}-cuda11-archive.tar.xz"
wget "https://developer.download.nvidia.com/compute/machine-learning/cudnn/v${CUDNN_VERSION}/${CUDNN_TARBALL}"
tar -xvf ${CUDNN_TARBALL}
sudo cp -P cuda/include/* /usr/local/cuda/include/
sudo cp -P cuda/lib64/* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/* /usr/local/cuda/lib64/*

# Install TensorFlow in Python Virtual Environment
echo "Setting up Python virtual environment and installing TensorFlow..."
TF_VERSION="2.10.0"
PYTHON_ENV="tf-env"

python3 -m venv ${PYTHON_ENV}
source ${PYTHON_ENV}/bin/activate

pip install --upgrade pip
pip install tensorflow==${TF_VERSION}

# Verify TensorFlow GPU Support
echo "Verifying TensorFlow GPU support..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"

echo "Setup completed! TensorFlow with GPU support for Tesla P4 is ready."
