
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install --no-install-recommends -y curl && apt-get -y install git

RUN apt-get update -y && \
    apt-get install -y python3-pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --pre --extra-index-url https://download.hidet.org/whl hidet
RUN pip install hidet numpy==1.23.0
RUN pip install diffusers transformers accelerate xformers

WORKDIR /code

COPY test_hidet.py /code/test_hidet.py

CMD [ "python3", "/code/test_hidet.py"]
