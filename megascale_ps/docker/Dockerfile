FROM nvidia/cuda:10.0-devel-ubuntu18.04

ARG https_proxy
ARG http_proxy

ARG MEGASCALE_PS_BASE_PATH=/usr/local
ARG MEGASCALE_PS_PATH=$MEGASCALE_PS_BASE_PATH/megascale_ps
ARG MEGASCALE_PS_GIT_LINK=https://github.com/bytedance/megascale_ps
ARG MEGASCALE_PS_BRANCH=master

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        tzdata \
        ca-certificates \
        git \
        curl \
        wget \
        vim \
        cmake \
        lsb-release \
        libcudnn7=7.6.0.64-1+cuda10.0 \
        libnuma-dev \
        ibverbs-providers \
        librdmacm-dev \
        ibverbs-utils \
        rdmacm-utils \
        libibverbs-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        libnccl2=2.4.7-1+cuda10.0 \
        libnccl-dev=2.4.7-1+cuda10.0

# install framework
# note: for tf <= 1.14, you need gcc-4.9
ARG FRAMEWORK=tensorflow
RUN if [ "$FRAMEWORK" = "tensorflow" ]; then \
        pip3 install --upgrade pip; \
        pip3 install -U tensorflow-gpu==1.15.0; \
    elif [ "$FRAMEWORK" = "pytorch" ]; then \
        pip3 install -U numpy==1.18.1 torchvision==0.5.0 torch==1.4.0; \
    elif [ "$FRAMEWORK" = "mxnet" ]; then \
        pip3 install -U mxnet-cu100==1.5.0; \
    else \
        echo "unknown framework: $FRAMEWORK"; \
        exit 1; \
    fi

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

RUN cd $MEGASCALE_PS_BASE_PATH &&\
    git clone --recursive -b $MEGASCALE_PS_BRANCH $MEGASCALE_PS_GIT_LINK &&\
    cd $MEGASCALE_PS_PATH &&\
    python3 setup.py install
