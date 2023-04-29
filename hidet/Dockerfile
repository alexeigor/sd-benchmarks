
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install --no-install-recommends -y curl && apt-get -y install git

RUN apt-get update -y && \
    apt-get install -y python3-pip

ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

# RUN python3 -m pip --no-cache-dir install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --pre --extra-index-url https://download.hidet.org/whl hidet
RUN pip install hidet numpy==1.23.0
RUN pip install diffusers transformers accelerate xformers

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY test_hidet.py /code/test_hidet.py

CMD [ "python", "/code/test_hidet.py"]
