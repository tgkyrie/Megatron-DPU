# BytePS 基础阶段：构建并安装 BytePS，可独立作为 byteps-only 镜像
# 默认使用 PyTorch + CUDA 11.8 + cuDNN8 开发镜像，带完整 CUDA 头文件便于编译插件
ARG BASE_IMAGE=pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel
ARG BYTEPS_IMAGE=byteps
FROM ${BASE_IMAGE} AS byteps

ARG http_proxy
ARG https_proxy

ARG BYTEPS_BASE_PATH=/usr/local
ARG BYTEPS_PATH=$BYTEPS_BASE_PATH/byteps
ARG BYTEPS_GIT_LINK=https://github.com/bytedance/byteps
ARG BYTEPS_BRANCH=master

ENV DEBIAN_FRONTEND=noninteractive

# 运行时和构建依赖，含 numactl/libnuma 及 RDMA 组件
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates git curl vim wget cmake build-essential \
    numactl libnuma-dev \
    ibverbs-providers librdmacm-dev ibverbs-utils rdmacm-utils libibverbs-dev \
    python3-dev python3-pip python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U cloudpickle==3.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR ${BYTEPS_BASE_PATH}
# 将本地仓库中的 byteps 目录（Megatron-DPU/byteps）作为构建源
COPY ./byteps/ ${BYTEPS_BASE_PATH}/byteps/
# 便捷脚本放入 /usr/local，进入容器即可直接调用（路径需显式指定或自行加 PATH）
COPY ./byteps/sh/server.sh ./byteps/sh/scheduler.sh ./byteps/sh/worker.sh /usr/local/
RUN chmod +x /usr/local/server.sh /usr/local/scheduler.sh /usr/local/worker.sh
RUN cd $BYTEPS_PATH && python3 setup.py install

# 便捷：把 bpslaunch 加入 PATH（通常 setup.py 已安装）
ENV PATH="/usr/local/bin:${PATH}"

CMD ["bash"]


# Megatron 阶段：在已构建好的 BytePS 镜像上增加 Megatron-LM、apex 等
# 默认使用当前构建的 byteps 阶段；在 CI 中可通过 BYTEPS_IMAGE 覆盖为已推送的 byteps 镜像
FROM ${BYTEPS_IMAGE} AS megatron

ARG BYTEPS_BASE_PATH=/usr/local
ARG MEGA_PATH=$BYTEPS_BASE_PATH/Megatron-LM

ENV UV_PROJECT_ENVIRONMENT=./venv
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:/root/.local/bin:$PATH"
ENV UV_LINK_MODE=copy

WORKDIR ${BYTEPS_BASE_PATH}
# 先复制变化较小的安装脚本，再安装基础构建依赖，减少源码变动对缓存的影响
COPY ./uvinstall.sh ${BYTEPS_BASE_PATH}
RUN chmod +x ${BYTEPS_BASE_PATH}/uvinstall.sh && ${BYTEPS_BASE_PATH}/uvinstall.sh
# 基础 Python 构建工具，前置安装以便缓存复用
RUN pip install -U pip setuptools wheel packaging ninja pybind11 -i https://mirrors.ustc.edu.cn/pypi/simple

# Megatron 源码拷贝放在后面，减少对前面层的缓存失效
WORKDIR ${MEGA_PATH}
COPY ./Megatron-LM/ ${BYTEPS_BASE_PATH}/Megatron-LM/
RUN pip install --no-build-isolation -e .
# GPU 服务器上安装 Megatron-Core MLM 额外依赖
RUN pip install -U "megatron-core[mlm]" -i https://mirrors.ustc.edu.cn/pypi/simple

RUN git clone --depth 1 https://github.com/NVIDIA/apex.git ${BYTEPS_BASE_PATH}/apex
COPY ./vocab/ ${BYTEPS_BASE_PATH}/vocab/

WORKDIR ${BYTEPS_BASE_PATH}
RUN cd apex && python setup.py install --cuda_ext --cpp_ext

CMD ["bash"]
