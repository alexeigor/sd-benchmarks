FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y curl && \
    apt-get -y install git && \
    apt-get install -y python3-pip && \
    pip install --upgrade pip

RUN pip install tensorrt
    
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

WORKDIR /code

RUN git clone https://github.com/NVIDIA/TensorRT.git

WORKDIR /code/TensorRT/demo/Diffusion/

RUN pip install -r ./requirements.txt
RUN mkdir -p onnx engine output

CMD cd /code/TensorRT/demo/Diffusion && python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom"


## RUN pip install onnxruntime-gpu>=1.14 py3nvml xformers

## COPY benchmark.py /code/benchmark.py

## CMD python3 benchmark.py --engine torch --batch_size 1 --batch_count 5
