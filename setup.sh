#!/bin/bash 

# Exit on error
set -e

echo "=== Starting QuantumDAGKnight Node Setup ==="

# Function to display setup progress
setup_status() {
    echo -e "\n=== $1 ===\n"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root with: sudo $0"
    exit 1
fi

# Update system
setup_status "Updating system packages"
apt update && apt upgrade -y

# Install basic dependencies
setup_status "Installing basic dependencies"
apt install -y \
    cmake \
    libssl-dev \
    build-essential \
    libbz2-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    ethtool \
    net-tools \
    sysstat \
    htop \
    iftop \
    redis-server \
    libsystemd-dev

# Install MongoDB
setup_status "Installing MongoDB"
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb.gpg
echo "deb [signed-by=/usr/share/keyrings/mongodb.gpg] https://repo.mongodb.org/apt/debian bullseye/mongodb-org/6.0 main" | tee /etc/apt/sources.list.d/mongodb-org-6.0.list
apt update
apt install -y mongodb-org

# Create the required directory for MongoDB PID file
mkdir -p /run/mongodb
chown mongodb:mongodb /run/mongodb

# Enable and start the MongoDB service
systemctl start mongod
systemctl enable mongod

# Install PostgreSQL
setup_status "Installing PostgreSQL"
apt install -y postgresql postgresql-contrib
systemctl start postgresql
systemctl enable postgresql

# Set up PostgreSQL
setup_status "Configuring PostgreSQL"
DB_USER="quantum_user"
DB_PASSWORD=$(openssl rand -hex 16)

# Check if the role already exists
if sudo -u postgres psql -t -c "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1; then
    echo "Role $DB_USER already exists. Skipping creation."
else
    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
fi

sudo -u postgres psql -c "ALTER ROLE $DB_USER SET client_encoding TO 'utf8';"
sudo -u postgres psql -c "ALTER ROLE $DB_USER SET default_transaction_isolation TO 'read committed';"
sudo -u postgres psql -c "ALTER ROLE $DB_USER SET timezone TO 'UTC';"

# Check if the database already exists
if sudo -u postgres psql -t -c "SELECT 1 FROM pg_database WHERE datname='quantum_dagknight'" | grep -q 1; then
    echo "Database 'quantum_dagknight' already exists. Skipping creation."
else
    sudo -u postgres psql -c "CREATE DATABASE quantum_dagknight;"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE quantum_dagknight TO $DB_USER;"
fi

# Set up virtual environment
setup_status "Setting up Python virtual environment"
python3 -m venv /home/myuser/QuantumDagknightCoin/venv2992
source /home/myuser/QuantumDagknightCoin/venv2992/bin/activate

# Install liboqs and Python wrapper in virtual environment
setup_status "Installing quantum cryptography libraries"
if [ ! -d "liboqs-python" ]; then
    git clone --recursive https://github.com/open-quantum-safe/liboqs-python.git
    cd liboqs-python
    python3 setup.py build
    python3 setup.py install
    cd ..
    rm -rf liboqs-python
else
    echo "liboqs-python already exists. Skipping installation."
fi

# Set up network optimizations
setup_status "Configuring network optimizations"
cat > /etc/sysctl.d/99-quantum-dagknight.conf << EOL
# Network buffer tuning
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 1048576
net.core.wmem_default = 1048576
net.ipv4.tcp_rmem = 4096 1048576 16777216
net.ipv4.tcp_wmem = 4096 1048576 16777216

# Connection handling
net.ipv4.tcp_max_syn_backlog = 8192
net.core.somaxconn = 8192
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_fin_timeout = 30

# TCP optimization
net.ipv4.tcp_slow_start_after_idle = 0
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_mtu_probing = 1

# UDP optimization for quantum state sync
net.ipv4.udp_rmem_min = 8192
net.ipv4.udp_wmem_min = 8192

# File descriptors
fs.file-max = 1000000
EOL

# Apply sysctl settings
sysctl -p /etc/sysctl.d/99-quantum-dagknight.conf

# Set up system limits for system performance
setup_status "Configuring system limits"
cat > /etc/security/limits.d/quantum-dagknight.conf << EOL
*               soft    nofile          65535
*               hard    nofile          65535
*               soft    nproc           32768
*               hard    nproc           32768
*               soft    memlock         unlimited
*               hard    memlock         unlimited
EOL

# Set up systemd service
setup_status "Creating systemd service"
cat > /etc/systemd/system/quantum-dagknight.service << EOL
[Unit]
Description=QuantumDAGKnight P2P Node Service
After=network.target redis.service mongodb.service postgresql.service
Wants=redis.service mongodb.service postgresql.service
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=notify
WorkingDirectory=/home/myuser/QuantumDagknightCoin
Environment=PYTHONPATH=/home/myuser/QuantumDagknightCoin
Environment=LOG_LEVEL=INFO
Environment=P2P_HOST=0.0.0.0
Environment=P2P_PORT=8765
Environment=MAX_PEERS=50
Environment=DB_USER=${DB_USER}
Environment=DB_PASSWORD=${DB_PASSWORD}
Environment=DB_HOST=localhost
Environment=DB_NAME=quantum_dagknight

ExecStart=/home/myuser/QuantumDagknightCoin/venv2992/bin/python3 /home/myuser/QuantumDagknightCoin/quantumdagknight.py

# Tuning parameters
LimitNOFILE=65535
LimitNPROC=4096
TasksMax=4096

# Resource limits
CPUQuota=85%
MemoryMax=4G
IOWeight=500

# Restart configuration
Restart=always
RestartSec=30s
TimeoutStartSec=120
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable quantum-dagknight

# Install Python dependencies in virtual environment
setup_status "Installing Python dependencies"
source /home/myuser/QuantumDagknightCoin/venv2992/bin/activate
pip install -r requirements.txt systemd-python

# Create configuration file
setup_status "Creating configuration file"
mkdir -p /home/myuser/QuantumDagknightCoin/config
cat > /home/myuser/QuantumDagknightCoin/config/config.yaml << EOL
node:
  host: 0.0.0.0
  port: 8765
  max_peers: 50
  bootstrap_nodes:
    - "node1.example.com:8765"
    - "node2.example.com:8765"

database:
  user: ${DB_USER}
  password: ${DB_PASSWORD}
  host: localhost
  name: quantum_dagknight

logging:
  level: INFO
  file: /var/log/quantum_dagknight/node.log

quantum:
  security_level: 20
  decoherence_threshold: 0.8
  heartbeat_interval: 5
EOL
cat >> /home/myuser/QuantumDagknightCoin/config/config.yaml << EOL

network_optimizer:
  enabled: true
  metrics_interval: 5
  optimization_interval: 30
  thresholds:
    latency_ms: 100
    packet_loss: 0.01
    bandwidth_mbps: 10
EOL
# Set up logging directory with appropriate permissions
mkdir -p /var/log/quantum_dagknight
chmod 755 /var/log/quantum_dagknight

# Set correct permissions for project files
chmod 755 /home/myuser/QuantumDagknightCoin
chmod +x /home/myuser/QuantumDagknightCoin/quantumdagknight.py
chmod -R 755 /home/myuser/QuantumDagknightCoin/venv2992/bin

# Start services
setup_status "Starting services"
systemctl start redis-server
systemctl start mongod
systemctl start postgresql
systemctl start quantum-dagknight

# Display completion message and important information
cat << EOL

=== QuantumDAGKnight Node Setup Complete ===

Important Information:
- Virtual Environment: /home/myuser/QuantumDagknightCoin/venv2992
- Database User: ${DB_USER}
- Database Password: ${DB_PASSWORD}
- Configuration File: /home/myuser/QuantumDagknightCoin/config/config.yaml
- Log File: /var/log/quantum_dagknight/node.log
- Service Status: systemctl status quantum-dagknight
- Service Logs: journalctl -u quantum-dagknight

Node Management Commands:
- Start node:   sudo systemctl start quantum-dagknight
- Stop node:    sudo systemctl stop quantum-dagknight
- Restart node: sudo systemctl restart quantum-dagknight
- View logs:    sudo journalctl -u quantum-dagknight -f

Virtual Environment Commands:
- Activate: source /home/myuser/QuantumDagknightCoin/venv2992/bin/activate
- Install packages: pip install <package-name>

Please update the bootstrap_nodes in the config file with actual node addresses.
Remember to save the database credentials securely.

EOL

# Final status check
systemctl is-active --quiet quantum-dagknight && echo "Node service is running" || echo "Node service failed to start"
