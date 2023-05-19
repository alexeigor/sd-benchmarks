FROM nvcr.io/nvidia/pytorch:23.02-py3
# custom torch 1.14 build that supports H100

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y git curl python3-pip

WORKDIR /code

RUN pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

RUN git clone https://github.com/microsoft/onnxruntime
RUN pip install -r /code/onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion/requirements-cuda.txt

RUN curl https://raw.githubusercontent.com/huggingface/diffusers/v0.15.1/scripts/convert_stable_diffusion_checkpoint_to_onnx.py > /code/convert_sd_onnx.py

CMD python3 /code/convert_sd_onnx.py --model_path runwayml/stable-diffusion-v1-5  --output_path  ./sd_v1_5/fp32 && \
    python3 /code/onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion/optimize_pipeline.py -i ./sd_v1_5/fp32 -o ./sd_v1_5/fp16 --float16 && \
    python3 /code/onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion/benchmark.py -p ./sd_v1_5/fp16 -b 1 -v 1.5

