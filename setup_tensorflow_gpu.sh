#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting setup for TensorFlow with GPU (Tesla P4) on Debian..."

# Update and upgrade the system
echo "Updating and upgrading the system..."
sudo apt update && sudo apt upgrade -y

# Install prerequisites
echo "Installing prerequisites..."
sudo apt install -y build-essential gcc make git curl wget software-properties-common python3 python3-pip python3-venv

# Add NVIDIA repository and install drivers
echo "Adding NVIDIA repository and installing GPU drivers..."
sudo apt install -y nvidia-detect
NVIDIA_DRIVER=$(nvidia-detect | grep -oP 'driver: \K.*')

if [ -z "$NVIDIA_DRIVER" ]; then
  echo "Failed to detect NVIDIA driver. Please check your hardware compatibility."
  exit 1
fi

# Install the recommended NVIDIA driver
sudo apt install -y nvidia-driver

# Reboot might be necessary for the driver to be fully operational
echo "Please reboot if this is the first time installing NVIDIA drivers."

# Verify NVIDIA installation
echo "Verifying NVIDIA installation..."
if ! nvidia-smi; then
  echo "nvidia-smi command failed. Ensure the GPU is properly installed and drivers are loaded."
  exit 1
fi

# Install CUDA Toolkit
echo "Installing CUDA Toolkit..."
CUDA_VERSION="11.8"
CUDA_DEB="cuda-repo-debian12-${CUDA_VERSION}-local_${CUDA_VERSION}.deb"
wget "https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_DEB}"
sudo dpkg -i ${CUDA_DEB}
sudo apt-key add /var/cuda-repo-debian12-${CUDA_VERSION}-local/7fa2af80.pub
sudo apt update
sudo apt install -y cuda

# Set environment variables
echo "Configuring environment variables for CUDA..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
echo "Verifying CUDA installation..."
if ! nvcc --version; then
  echo "CUDA installation verification failed. Ensure the CUDA Toolkit is correctly installed."
  exit 1
fi

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
