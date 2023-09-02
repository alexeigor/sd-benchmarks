FROM nvcr.io/nvidia/pytorch:23.08-py3
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html

RUN apt-get update && apt-get install -y \
    && apt-get install --no-install-recommends -y git curl python3-pip \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /code

RUN pip install diffusers transformers accelerate

COPY test_pt.py /code/test_pt.py

CMD python3 /code/test_pt.py
