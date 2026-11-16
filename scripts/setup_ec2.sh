#!/usr/bin/env bash
set -euo pipefail

# EC2 Ubuntu Server Bootstrap Script
echo "=== Updating packages and installing dependencies ==="
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git tmux htop ufw

# Set timezone to Asia/Kolkata
echo "=== Setting server timezone to Asia/Kolkata ==="
sudo timedatectl set-timezone Asia/Kolkata

# Optional swap file for small instances (t3.micro/t3.small)
if [ ! -f /swapfile ]; then
    echo "=== Creating 2GB swap file ==="
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Setup Python virtual environment
echo "=== Setting up Python environment ==="
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup log directory
mkdir -p "$HOME/logs"

# Install systemd service
echo "=== Installing systemd dashboard service ==="
sudo cp deploy/aqi-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable aqi-dashboard
sudo systemctl restart aqi-dashboard

# Install crontab
echo "=== Installing cron schedules ==="
crontab deploy/crontab.txt

echo "=== Setup complete! Dashboard is running at http://$(curl -s http://checkip.amazonaws.com):8501 ==="
