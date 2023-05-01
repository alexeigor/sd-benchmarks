FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y curl && \
    apt-get -y install git && \
    apt-get install -y python3-pip
    
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

WORKDIR /code

RUN pip install diffusers==0.15.1 transformers==4.26.0 numpy==1.24.1 accelerate==0.15.0 \
    onnx==1.13.0 coloredlogs packaging==23.0 protobuf==3.20.3 psutil==5.9.4 sympy==1.11.1

RUN pip install onnxruntime-gpu>=1.14 py3nvml xformers

COPY benchmark.py /code/benchmark.py

CMD python3 benchmark.py --engine torch --batch_size 1 --batch_count 5
