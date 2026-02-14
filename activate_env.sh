#!/bin/bash
# Activate the adcc development environment
# Usage: source activate_env.sh

export LD_LIBRARY_PATH="/home/scratch.cuc_gpu_1/adcc/libtensorlight/lib:/home/scratch.cuc_gpu_1/adcc/openblas_install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="/home/scratch.cuc_gpu_1/adcc/libtensorlight/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
source /home/scratch.cuc_gpu_1/adcc/venv_adcc/bin/activate
echo "adcc dev environment activated (venv + libtensorlight + openblas)"
