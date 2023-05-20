FROM nvcr.io/nvidia/pytorch:23.04-py3
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y git curl python3-dev python3-pip && \
    pip install --upgrade pip
    
# RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118

WORKDIR /code

RUN pip install git+https://github.com/huggingface/diffusers transformers accelerate

COPY test_pt.py /code/test_pt.py

CMD python3 /code/test_pt.py
