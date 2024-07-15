#!/bin/bash

cd ~

export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc

sudo NEEDRESTART_MODE=a apt-get update -y
sudo NEEDRESTART_MODE=a apt-get install -y python3.10-venv gcsfuse

mkdir -p ~/rosinality-tpu-bucket
gcsfuse -o rw --implicit-dirs --http-client-timeout=5s --max-conns-per-host=2000 \
        --debug_fuse_errors --debug_fuse --debug_gcs --debug_invariants --debug_mutex \
        --log-file=$HOME/gcsfuse.json "rosinality-tpu-bucket" "~/rosinality-tpu-bucket"

python3.10 -m venv dev

source dev/bin/activate
python3 -m pip install -U pip setuptools wheel ipython
python3 -m pip install --upgrade pip

cd ~/nanodo
pip install -e .

pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html