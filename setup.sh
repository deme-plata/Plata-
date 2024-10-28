#!/bin/bash

# Update and install MongoDB
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/debian buster/mongodb-org/6.0 main" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

sudo apt update
sudo apt-get install -y mongodb-org

# Start MongoDB and enable it to start on boot
sudo systemctl start mongod
sudo systemctl enable mongod

# Install required packages for building liboqs
sudo apt-get install -y cmake libssl-dev build-essential

# Clone, build, and install liboqs and Python wrapper
git clone --recursive https://github.com/open-quantum-safe/liboqs-python.git
cd liboqs-python
python3 setup.py build
python3 setup.py install
cd ..

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Run the DAGKnight app
python quantumdagknight.py
