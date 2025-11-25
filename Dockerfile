# 基础镜像已包含 CUDA 11.8、cuDNN8、NCCL、PyTorch 2.2.2（cu118）
FROM byteps-cu118:pt22

ARG http_proxy
ARG https_proxy

# 可按需调整
ARG BYTEPS_BASE_PATH=/usr/local
ARG BYTEPS_PATH=$BYTEPS_BASE_PATH/byteps
ARG BYTEPS_GIT_LINK=https://github.com/bytedance/byteps
ARG BYTEPS_BRANCH=master

ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U cloudpickle==3.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# RDMA & 构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates git curl vim wget cmake build-essential \
    libnuma-dev \
    ibverbs-providers librdmacm-dev ibverbs-utils rdmacm-utils libibverbs-dev \
    python3-dev python3-pip python3-setuptools \
 && rm -rf /var/lib/apt/lists/*

# 可选：升级 pip
# RUN python3 -m pip install -U pip

# （可选）装一个较新 NCCL 工具链；pytorch 镜像里已经带运行时，一般不用动
# RUN python3 -m pip install -U nvidia-nccl-cu11==2.18.*  # 仅当需要

# 编译安装 BytePS（按官方步骤）

WORKDIR ${BYTEPS_BASE_PATH}
# 注意：COPY 的源路径是“构建上下文”里的相对路径
COPY ./byteps/ ${BYTEPS_BASE_PATH}/byteps/



RUN cd $BYTEPS_PATH \
 && python3 setup.py install


# RUN cd $BYTEPS_BASE_PATH \
# #  && git clone --recursive -b $BYTEPS_BRANCH $BYTEPS_GIT_LINK \
#  && cd $BYTEPS_PATH \
#  && python3 setup.py install

# 便捷：把 bpslaunch 加入 PATH（通常 setup.py 已安装）
ENV PATH="/usr/local/bin:${PATH}"

# 默认使用 python3
CMD ["bash"]


ARG MEGA_PATH=$BYTEPS_BASE_PATH/Megatron-LM

COPY ./Megatron-LM/ ${BYTEPS_BASE_PATH}/Megatron-LM/
COPY ./uvinstall.sh ${BYTEPS_BASE_PATH}

# 安装 uv（按 Megatron 文档）
RUN cd $BYTEPS_BASE_PATH
RUN chmod +x uvinstall.sh
RUN sh uvinstall.sh


ENV UV_PROJECT_ENVIRONMENT=./venv
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:/root/.local/bin:$PATH"
ENV UV_LINK_MODE=copy

WORKDIR ${MEGA_PATH}

RUN pip install -U pip setuptools wheel packaging ninja pybind11 -i https://mirrors.ustc.edu.cn/pypi/simple
# RUN pip install --no-build-isolation -U "megatron-core[dev,mlm]" -i https://mirrors.ustc.edu.cn/pypi/simple 
RUN pip install --no-build-isolation -e .

# in GPU server run this
RUN pip install -U "megatron-core[mlm]" -i https://mirrors.ustc.edu.cn/pypi/simple 

COPY ./apex/ ${BYTEPS_BASE_PATH}/apex/ 
COPY ./vocab/ ${BYTEPS_BASE_PATH}/apex/
 
WORKDIR ${BYTEPS_BASE_PATH}
RUN cd apex && python setup.py install --cuda_ext --cpp_ext