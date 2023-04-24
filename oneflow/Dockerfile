
FROM ubuntu:latest

RUN apt-get update && apt-get install --no-install-recommends -y curl && apt-get -y install git
RUN apt-get install -y python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade pip

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117
RUN python3 -m pip install "torch<2" "transformers>=4.26" "diffusers[torch]==0.14.0"
RUN python3 -m pip uninstall accelerate -y
RUN python3 -m pip install -U onediff

COPY test_oneflow.py /code/test_oneflow.py

CMD [ "python3", "/code/test_oneflow.py"]
