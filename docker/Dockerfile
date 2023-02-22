
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:23.01-py3
    
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        autogen \
        clangd \
        gdb \
        git-lfs \
        libb64-dev \
        libcusparse-dev-11-8 \
        libz-dev \
        locales-all \
        mosh \
        openssh-server \
        python3-dev \
        rapidjson-dev \
        sudo \
        tmux \
        unzip \
        zstd \
        zip \
        zsh \
        python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && pip3 install --upgrade requests && \
    pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-1.14.0a0+44dac51c.nv23.01-cp38-cp38-linux_aarch64.whl && \
    pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v51 tensorflow==2.11.0+nv23.01 && \
    pip3 install --extra-index-url https://pypi.ngc.nvidia.com regex \
                    fire tritonclient[all] && \
    pip3 install transformers huggingface_hub tokenizers SentencePiece \
                    sacrebleu datasets tqdm omegaconf rouge_score jupyterlab && \
    pip3 install cmake==3.24.3

# backend build
ADD . /workspace/build/fastertransformer_backend
RUN mkdir -p /workspace/build/fastertransformer_backend/build
WORKDIR /workspace/build/fastertransformer_backend/build
RUN cmake \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      ..
RUN make -j"$(grep -c ^processor /proc/cpuinfo)" install
    
ENV WORKSPACE /workspace
WORKDIR /workspace       

ARG SM=80
ARG FORCE_BACKEND_REBUILD=0
ARG ENABLE_FP8=OFF
RUN git clone https://github.com/NVIDIA/FasterTransformer.git && mkdir -p FasterTransformer/build && \
    cd FasterTransformer && git submodule update --init --recursive && cd build && \
    cmake -DSM=${SM} -DCMAKE_BUILD_TYPE=Release -DBUILD_TF2=ON -DTF_PATH=/usr/local/lib/python3.8/dist-packages/tensorflow/ -DENABLE_FP8=${ENABLE_FP8} .. && \
    make -j"$(grep -c ^processor /proc/cpuinfo)"
ENV NCCL_LAUNCH_MODE=GROUP
RUN sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config \
    && mkdir /var/run/sshd -p
