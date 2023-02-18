# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG TRITON_VERSION=22.12
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE}

# apt-get install -y cuda-toolkit cuda-tools cuda-cupti libopenblas-dev cargo
RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/3bf863cc.pub
# RUN apt-get update -q
RUN apt-get install -y cuda-compat-12-0 && apt-get install -y --no-install-recommends \
       autoconf autogen clangd gdb git-lfs libb64-dev libz-dev locales-all mosh openssh-server python3-dev rapidjson-dev sudo tmux unzip cargo  zstd zip zsh && \
    apt-get install -y cuda-toolkit-11-4 cuda-tools-11-4 cuda-cupti-11-4 libopenblas-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip3 install -f torch==1.9.1+cu111 torchvision && \
#    pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl && \
#pip3 install torch==1.13.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html && \
RUN pip3 install --upgrade pip
# && \
#    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 && \
#    pip3 install --extra-index-url https://pypi.ngc.nvidia.com regex fire tritonclient[all] && \
#    pip3 install transformers huggingface_hub tokenizers SentencePiece sacrebleu datasets tqdm omegaconf rouge_score && \
#    pip3 install cmake==3.24.3

# Below is good, jus needs torch testing first...
# backend build
#ADD . /workspace/build/fastertransformer_backend
#RUN mkdir -p /workspace/build/fastertransformer_backend/build

#WORKDIR /workspace/build/fastertransformer_backend/build
#ARG FORCE_BACKEND_REBUILD=0
#RUN cmake \
#      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
#      -D CMAKE_BUILD_TYPE=Release \
#      -D ENABLE_FP8=OFF \
#      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
#      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
#      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
#      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" ..

#RUN cd _deps/repo-ft-src/ && \
#    git log | head -n 3 2>&1 | tee /workspace/build/fastertransformer_backend/FT_version.txt && \
#    cd /workspace/build/fastertransformer_backend/build && \
#    make -j"$(grep -c ^processor /proc/cpuinfo)" install && \
#    rm /workspace/build/fastertransformer_backend/build/bin/*_example -rf && \
#    rm /workspace/build/fastertransformer_backend/build/lib/lib*Backend.so -rf

ENV NCCL_LAUNCH_MODE=GROUP
ENV WORKSPACE /workspace
WORKDIR /workspace

RUN sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config && \
    mkdir /var/run/sshd -p
