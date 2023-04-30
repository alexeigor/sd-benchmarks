
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

RUN apt-get update && apt-get install --no-install-recommends -y curl && apt-get -y install git

RUN apt-get update -y && \
    apt-get install -y python3 python3-dev python3-pip

RUN git clone --recursive https://github.com/facebookincubator/AITemplate

RUN bash /AITemplate/docker/install/install_ait.sh

COPY test_ait.py /test_ait.py

CMD [ "python3", "/test_ait.py"]
