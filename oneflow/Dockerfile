FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /code

RUN pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu117
# RUN pip install -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu117 oneflow

RUN python3 -m pip install "torch" "transformers==4.27.1" "diffusers[torch]==0.19.3" && \
    python3 -m pip uninstall accelerate -y && \
    git clone https://github.com/Oneflow-Inc/diffusers.git onediff && \
    python3 -m pip install -e /code/onediff

COPY test_oneflow.py /code/test_oneflow.py

CMD [ "python3", "/code/test_oneflow.py"]
