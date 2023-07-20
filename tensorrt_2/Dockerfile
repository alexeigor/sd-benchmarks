FROM nvcr.io/nvidia/pytorch:23.06-py3
# custom torch 1.14 build that supports H100

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y git curl python3-pip && \
    pip install --upgrade pip

WORKDIR /code

RUN pip install diffusers transformers accelerate colored
RUN pip install --upgrade tensorrt>=8.6.1
RUN pip install --upgrade polygraphy>=0.47.0 onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
RUN pip install --upgrade onnxruntime

COPY test_trt.py /code/test_trt.py

CMD [ "python3", "/code/test_trt.py"]
