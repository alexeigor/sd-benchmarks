FROM nvcr.io/nvidia/pytorch:23.10-py3
# custom torch 1.14 build that supports H100

RUN apt-get update && apt-get install -y \
    && apt-get install --no-install-recommends -y git curl python3-pip \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN git clone https://github.com/microsoft/onnxruntime

WORKDIR /code/onnxruntime

RUN export CUDACXX=/usr/local/cuda-12.2/bin/nvcc && \
    git config --global --add safe.directory '*' && \ 
    sh build.sh --config Release  --build_shared_lib --parallel --use_cuda --cuda_version 12.2 \
            --cuda_home /usr/local/cuda-12.2 --cudnn_home /usr/lib/x86_64-linux-gnu/ --build_wheel --skip_tests \
            --use_tensorrt --tensorrt_home /usr/src/tensorrt \
            --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=80 \
            --allow_running_as_root && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl --force-reinstall

WORKDIR /code/onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion

RUN python3 -m pip install -r requirements-cuda12.txt && \
    python3 -m pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com

CMD cd /code/onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion && \
    python3 demo_txt2img_xl.py --disable-refiner --denoising-steps 50 \
    --engine ORT_CUDA --work-dir /workspace/ "starry night over Golden Gate Bridge by van gogh"

# CMD cd /code/onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion && \
#     python3 demo_txt2img.py --engine ORT_CUDA --version 2.1 --height 768 --width 768 --batch-size 1 \
#     --disable-refiner --work-dir /workspace/ "starry night over Golden Gate Bridge by van gogh"