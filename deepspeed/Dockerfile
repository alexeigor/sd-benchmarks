FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt-get update && apt-get install -y \
    && apt-get install --no-install-recommends -y git curl python3-pip \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN pip install deepspeed diffusers transformers accelerate

COPY test_deepspeed.py /code/test_deepspeed.py

CMD [ "python3", "/code/test_deepspeed.py"]
