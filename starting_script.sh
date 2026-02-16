#!/bin/bash

# Update system & install essentials
sudo apt update && sudo apt install -y python3 python3-venv python3-pip tmux htop git build-essential unzip mdadm

# GPU setup (runs only if NVIDIA GPU detected)
if lspci | grep -i nvidia; then
  sudo apt install -y ubuntu-drivers-common
  sudo ubuntu-drivers autoinstall
  echo "GPU drivers installed. Please reboot the VM manually for changes to take effect."
fi

# Create and activate Python virtual environment
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# Upgrade pip inside venv
pip install --upgrade pip

# Install Python packages inside venv
pip install pandas dask[complete] modin[all] pyarrow tqdm numpy scipy scikit-learn matplotlib seaborn jupyterlab

# Local SSD RAID 0 setup
DISKS=("/dev/nvme0n1" "/dev/nvme0n2")
RAID_DEVICE="/dev/md0"
MOUNT_POINT="/mnt/disks/ssd"

# Check that both SSDs exist
if [ -b "${DISKS[0]}" ] && [ -b "${DISKS[1]}" ]; then
  echo "Creating RAID 0 array from ${DISKS[0]} and ${DISKS[1]}"
  
  # Stop any existing md0 to avoid errors
  sudo mdadm --stop $RAID_DEVICE 2>/dev/null || true
  sudo mdadm --zero-superblock --force ${DISKS[0]} ${DISKS[1]} 2>/dev/null || true

  # Create RAID 0
  yes | sudo mdadm --create --verbose $RAID_DEVICE --level=0 --raid-devices=2 ${DISKS[0]} ${DISKS[1]}
  
  # Wait for RAID to initialize
  sleep 10

  # Create mdadm config so RAID assembles on boot
  sudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf
  sudo update-initramfs -u

  # Format RAID device
  sudo mkfs.ext4 -F $RAID_DEVICE

  # Mount RAID device
  sudo mkdir -p $MOUNT_POINT
  sudo mount -o defaults,noatime $RAID_DEVICE $MOUNT_POINT
  sudo chmod a+w $MOUNT_POINT

  # Add to fstab for persistence
  echo "$RAID_DEVICE $MOUNT_POINT ext4 defaults,noatime,nofail 0 2" | sudo tee -a /etc/fstab

  echo "RAID 0 setup complete, mounted at $MOUNT_POINT"
else
  echo "One or both local SSDs not found: ${DISKS[0]} ${DISKS[1]}"
fi
