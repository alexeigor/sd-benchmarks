
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

RUN apt-get update && apt-get install --no-install-recommends -y curl && apt-get -y install git

ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

RUN python3 -m pip --no-cache-dir install --upgrade pip

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY test_deepspeed.py /code/test_deepspeed.py

CMD [ "python", "/code/test_deepspeed.py"]
