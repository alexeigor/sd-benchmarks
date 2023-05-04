FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y curl && \
    apt-get -y install git && \
    apt-get install -y python3-pip

RUN pip install torch --index-url https://download.pytorch.org/whl/cu118

WORKDIR /code

RUN pip install deepspeed triton==2.0.0.dev20221202 diffusers transformers accelerate

COPY test_deepspeed.py /code/test_deepspeed.py

CMD [ "python3", "/code/test_deepspeed.py"]
