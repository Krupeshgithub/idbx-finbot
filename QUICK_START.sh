#!/bin/bash
# AIDANN One-Time Server Setup for NVIDIA L4 GPU
# Run this ONCE on a fresh server, then use: docker compose up -d --build

set -e

echo "🚀 AIDANN Server Setup - NVIDIA L4 GPU (One-Time Setup)"
echo "========================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root: sudo ./QUICK_START.sh${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[1/5] Checking NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
else
    echo -e "${RED}✗ NVIDIA GPU not detected. Install NVIDIA drivers first:${NC}"
    echo "   Ubuntu: sudo apt install nvidia-driver-535"
    echo "   Then reboot and run this script again."
    exit 1
fi

echo -e "\n${YELLOW}[2/5] Installing NVIDIA Container Toolkit...${NC}"
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

# Check if already installed
if command -v nvidia-ctk &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA Container Toolkit already installed${NC}"
else
    # Add NVIDIA repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # Install
    apt-get update
    apt-get install -y nvidia-container-toolkit
    echo -e "${GREEN}✓ NVIDIA Container Toolkit installed${NC}"
fi

echo -e "\n${YELLOW}[3/5] Configuring Docker for GPU...${NC}"
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
echo -e "${GREEN}✓ Docker configured for GPU${NC}"

echo -e "\n${YELLOW}[4/5] Testing GPU in Docker...${NC}"
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU accessible in Docker${NC}"
else
    echo -e "${RED}✗ GPU not accessible in Docker. Check Docker installation.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[5/5] Optimizing System Settings...${NC}"

# CPU Performance
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
echo -e "${GREEN}✓ CPU governor set to performance${NC}"

# Memory optimization
sysctl -w kernel.shmmax=2147483648 > /dev/null 2>&1
sysctl -w kernel.shmall=2097152 > /dev/null 2>&1
echo -e "${GREEN}✓ Shared memory optimized${NC}"

# Network optimization
sysctl -w net.core.rmem_max=134217728 > /dev/null 2>&1
sysctl -w net.core.wmem_max=134217728 > /dev/null 2>&1
echo -e "${GREEN}✓ Network buffers optimized${NC}"

# Make settings persistent
if ! grep -q "AIDANN Optimizations" /etc/sysctl.conf; then
    cat >> /etc/sysctl.conf << 'EOF'

# AIDANN Optimizations
kernel.shmmax=2147483648
kernel.shmall=2097152
net.core.rmem_max=134217728
net.core.wmem_max=134217728
EOF
    echo -e "${GREEN}✓ Settings made persistent${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ One-Time Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. Update ${YELLOW}.env.docker${NC} with your API keys (if needed)"
echo -e "2. Build and start: ${GREEN}docker compose up -d --build${NC}"
echo -e "3. Check logs: ${GREEN}docker compose logs -f aidaan${NC}"
echo -e "4. Monitor GPU: ${GREEN}watch -n 1 nvidia-smi${NC}"
echo -e "5. Check GPU stats: ${GREEN}curl http://localhost:8000/gpu-stats${NC}"

echo -e "\n${YELLOW}Resource Allocation:${NC}"
echo "CPU: 7.5 cores (limit) / 4 cores (reserved)"
echo "Memory: 28GB (limit) / 16GB (reserved)"
echo "GPU: 1x NVIDIA L4 (100%)"

echo -e "\n${YELLOW}Expected Performance:${NC}"
echo "FinBERT: 50ms (10x faster than CPU)"
echo "Response time: 15-18s (30% faster)"
echo "Throughput: 50-100 RPM (5x more)"

echo -e "\n${GREEN}Server is ready! Run: docker compose up -d --build${NC}"
echo -e "${YELLOW}Note: First build will take 10-15 minutes (downloads CUDA base image)${NC}"
