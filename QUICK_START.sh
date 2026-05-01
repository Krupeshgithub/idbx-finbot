#!/bin/bash
# Fix NVIDIA Container Toolkit Installation
# Run this to fix the broken repository configuration

set -e

echo "🔧 Fixing NVIDIA Container Toolkit Installation"
echo "================================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root: sudo ./FIX_NVIDIA_TOOLKIT.sh${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[1/5] Cleaning up broken repository file...${NC}"
rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
echo -e "${GREEN}✓ Cleaned up${NC}"

echo -e "\n${YELLOW}[2/5] Adding NVIDIA GPG key...${NC}"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo -e "${GREEN}✓ GPG key added${NC}"

echo -e "\n${YELLOW}[3/5] Adding NVIDIA repository...${NC}"
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
echo -e "${GREEN}✓ Repository added${NC}"

echo -e "\n${YELLOW}[4/5] Updating package list...${NC}"
apt-get update
echo -e "${GREEN}✓ Package list updated${NC}"

echo -e "\n${YELLOW}[5/5] Installing NVIDIA Container Toolkit...${NC}"
apt-get install -y nvidia-container-toolkit
echo -e "${GREEN}✓ NVIDIA Container Toolkit installed${NC}"

echo -e "\n${YELLOW}Configuring Docker for GPU...${NC}"
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
echo -e "${GREEN}✓ Docker configured for GPU${NC}"

echo -e "\n${YELLOW}Testing GPU in Docker...${NC}"
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU accessible in Docker${NC}"
else
    echo -e "${RED}✗ GPU not accessible in Docker${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ NVIDIA Container Toolkit Fixed!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. Build and start: ${GREEN}docker compose up -d --build${NC}"
echo -e "2. Check logs: ${GREEN}docker compose logs -f aidaan${NC}"
echo -e "3. Monitor GPU: ${GREEN}watch -n 1 nvidia-smi${NC}"

echo -e "\n${GREEN}Ready to deploy! 🚀${NC}"
