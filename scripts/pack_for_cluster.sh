#!/bin/bash
#
# Pack SSPO project for 8x H100 cluster deployment
#
# Usage:
#   bash scripts/pack_for_cluster.sh --full          # Full package (code + data ~92G)
#   bash scripts/pack_for_cluster.sh --code-only     # Code only (~2G), data mounted at runtime

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PACKAGE_NAME="sspo_cluster_package"
OUTPUT_DIR="${PROJECT_ROOT}/packaged"
MODE="${1:-}"

show_help() {
    echo "=========================================="
    echo "SSPO Cluster Packaging Script"
    echo "=========================================="
    echo ""
    echo "Usage: bash scripts/pack_for_cluster.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --full         Full package (code + data ~92GB)"
    echo "  --code-only    Code only (~2GB), data mounted at runtime"
    echo "  --help         Show this help"
    echo ""
    echo "Output:"
    echo "  ${OUTPUT_DIR}/${PACKAGE_NAME}_full.tar.gz"
    echo "  ${OUTPUT_DIR}/${PACKAGE_NAME}_code.tar.gz"
    echo ""
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker not found. Please install Docker first."
        exit 1
    fi
    echo "Docker found: $(docker --version)"
}

create_dockerfile() {
    local dockerfile="${OUTPUT_DIR}/Dockerfile"
    local include_cache="${1:-false}"

    if [ "$include_cache" = "true" ]; then
        # Full mode - include cache
        cat > "$dockerfile" << 'DOCKERFILE_EOF'
# SSPO - Semi-Supervised Preference Optimization
# Base image for 8x H100 Cluster
# Using Chinese mirror for faster download
FROM docker.m.daocloud.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl vim tmux htop \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /workspace

# Install PyTorch (CUDA 12.4)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
COPY src/requirements.txt /workspace/src/requirements.txt
RUN pip install --no-cache-dir -r /workspace/src/requirements.txt

# Copy source code
COPY src/src_sspo /workspace/src/src_sspo
COPY src/data /workspace/src/data
COPY scripts /workspace/scripts
COPY configs /workspace/configs

ENV PYTHONPATH="/workspace/src/src_sspo:${PYTHONPATH:-}"

# Copy cache for --full mode
COPY cache /workspace/cache

RUN mkdir -p /workspace/saves /workspace/logs /workspace/results

CMD ["/bin/bash"]
DOCKERFILE_EOF
    else
        # Code-only mode - no cache
        cat > "$dockerfile" << 'DOCKERFILE_EOF'
# SSPO - Semi-Supervised Preference Optimization
# Base image for 8x H100 Cluster
# Using Chinese mirror for faster download
FROM docker.m.daocloud.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl vim tmux htop \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /workspace

# Install PyTorch (CUDA 12.4)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
COPY src/requirements.txt /workspace/src/requirements.txt
RUN pip install --no-cache-dir -r /workspace/src/requirements.txt

# Copy source code
COPY src/src_sspo /workspace/src/src_sspo
COPY src/data /workspace/src/data
COPY scripts /workspace/scripts
COPY configs /workspace/configs

ENV PYTHONPATH="/workspace/src/src_sspo:${PYTHONPATH:-}"

RUN mkdir -p /workspace/saves /workspace/logs /workspace/results

CMD ["/bin/bash"]
DOCKERFILE_EOF
    fi

    echo "Created: $dockerfile"
}

create_entrypoint() {
    local entrypoint="${OUTPUT_DIR}/entrypoint.sh"

    cat > "$entrypoint" << 'ENTRYPOINT_EOF'
#!/bin/bash
#
# SSPO Cluster Entrypoint

set -euo pipefail

echo "=========================================="
echo "SSPO - Semi-Supervised Preference Optimization"
echo "=========================================="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Link mounted data if available
[ -d "/mnt/cache" ] && [ ! -d "/workspace/cache" ] && ln -s /mnt/cache /workspace/cache
[ -d "/mnt/saves" ] && ln -s /mnt/saves /workspace/saves

if [ -f "/workspace/src/data/dataset_info.json" ]; then
    echo "Datasets:"
    python -c "import json; d=json.load(open('/workspace/src/data/dataset_info.json')); print(f'  {len(d)} registered')"
fi

echo ""
echo "Commands:"
echo "  bash scripts/train.sh <config>    # Train"
echo "  bash scripts/quick_validate.sh    # Quick test"
echo ""

[ $# -gt 0 ] && exec "$@" || exec /bin/bash
ENTRYPOINT_EOF

    chmod +x "$entrypoint"
    echo "Created: $entrypoint"
}

package_code_only() {
    echo "=========================================="
    echo "Packaging CODE ONLY (~2GB)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"
    create_dockerfile "false"
    create_entrypoint

    echo ""
    echo "Building Docker image..."
    docker build -f "${OUTPUT_DIR}/Dockerfile" -t sspo:code-only "${PROJECT_ROOT}"

    echo ""
    echo "Saving Docker image..."
    docker save sspo:code-only | gzip > "${OUTPUT_DIR}/${PACKAGE_NAME}_code.tar.gz"

    echo ""
    ls -lh "${OUTPUT_DIR}/${PACKAGE_NAME}_code.tar.gz"
}

package_full() {
    echo "=========================================="
    echo "Packaging FULL (~92GB with data)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"
    create_dockerfile "true"
    create_entrypoint

    echo ""
    echo "Building Docker image..."
    docker build -f "${OUTPUT_DIR}/Dockerfile" -t sspo:full "${PROJECT_ROOT}"

    echo ""
    echo "Saving Docker image..."
    docker save sspo:full | gzip > "${OUTPUT_DIR}/${PACKAGE_NAME}_full.tar.gz"

    echo ""
    ls -lh "${OUTPUT_DIR}/${PACKAGE_NAME}_full.tar.gz"
}

main() {
    echo "SSPO Cluster Packaging"
    echo ""

    check_docker
    mkdir -p "$OUTPUT_DIR"

    case "$MODE" in
        --code-only) package_code_only ;;
        --full) package_full ;;
        --help|-h) show_help ;;
        *) show_help ;;
    esac
}

main
