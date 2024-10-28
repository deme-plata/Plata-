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

# Install PostgreSQL
sudo apt-get install -y postgresql postgresql-contrib

# Start PostgreSQL and enable it to start on boot
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Set up PostgreSQL user and database (adjust username and password as needed)
sudo -u postgres psql -c "CREATE USER your_postgresql_username WITH PASSWORD 'your_postgresql_password';"
sudo -u postgres psql -c "ALTER ROLE your_postgresql_username SET client_encoding TO 'utf8';"
sudo -u postgres psql -c "ALTER ROLE your_postgresql_username SET default_transaction_isolation TO 'read committed';"
sudo -u postgres psql -c "ALTER ROLE your_postgresql_username SET timezone TO 'UTC';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE postgres TO your_postgresql_username;"

# Update environment variables
echo "DB_USER=your_postgresql_username" >> ~/.bashrc
echo "DB_PASSWORD=your_postgresql_password" >> ~/.bashrc
echo "DB_HOST=localhost" >> ~/.bashrc
source ~/.bashrc

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Run the DAGKnight app
python quantumdagknight.py
