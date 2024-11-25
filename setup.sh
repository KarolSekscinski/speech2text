#!/bin/bash

# Update and upgrade the system
echo "Updating and upgrading the system..."
sudo apt update && sudo apt upgrade -y

# Install required dependencies
echo "Installing required dependencies..."
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

# Download Python 3.10 source code
PYTHON_VERSION="3.10.0"
echo "Downloading Python $PYTHON_VERSION source code..."
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz

# Extract the tar file
echo "Extracting Python $PYTHON_VERSION source code..."
tar -xvf Python-$PYTHON_VERSION.tgz

# Build and install Python
echo "Building and installing Python $PYTHON_VERSION..."
cd Python-$PYTHON_VERSION
sudo ./configure --enable-optimizations
sudo make -j $(nproc)
sudo make altinstall

# Verify Python installation
echo "Verifying Python installation..."
python3.10 --version


# Create a Python virtual environment
echo "Creating a Python virtual environment..."
python3.10 -m venv my_venv

# Activate the virtual environment
echo "Activating the Python virtual environment..."
source my_venv/bin/activate

# Install required Python packages
echo "Installing Python packages from requirements.txt..."
cd speech2text
pip install -r requirements.txt

echo "Setup complete!"
