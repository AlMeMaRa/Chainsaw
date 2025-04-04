#!/bin/bash

# Update and install dependencies for pyenv
echo "Installing dependencies for pyenv..."
if [ -f /etc/arch-release ]; then
    sudo pacman -Syu --noconfirm pyenv # openssl zlib xz tk git
elif [ -f /etc/lsb-release ]; then
    sudo apt update -y && sudo apt install -y make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
        libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git
else
    echo "Unsupported OS. Please install pyenv dependencies manually."
    exit 1
fi

# Install pyenv
echo "Installing pyenv..."
if [ -f /etc/arch-release ]; then
    echo "Pyenv is installed via pacman on Arch."
else
    curl https://pyenv.run | bash
fi

# Add pyenv to shell
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Install Python 11
echo "Installing Python 11..."
pyenv install 3.11.0
pyenv global 3.11.0

# Create a virtual environment
echo "Creating a virtual environment..."
python -m venv venv
source venv/bin/activate

# Install required Python packages with CUDA support
echo "Installing required Python packages with CUDA support..."
pip install -q tensorflow[and-cuda]==2.19.0 tensorflow-io==0.37.1 matplotlib kagglehub kaggle

echo "Setup complete. Activate the virtual environment using 'source venv/bin/activate'."
