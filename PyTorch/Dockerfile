FROM nvcr.io/nvidia/pytorch:24.05-py3
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html

RUN apt-get update && apt-get install -y \
    && apt-get install --no-install-recommends -y git curl python3-pip \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN
ENV HF_HOME="/workspace/.cache/huggingface"
    
WORKDIR /code

RUN pip install diffusers transformers accelerate sentencepiece

COPY test_pt.py /code/test_pt.py
COPY run_sd3_compile.py /code/run_sd3_compile.py

# CMD python3 /code/test_pt.py
CMD python3 /code/run_sd3_compile.py
