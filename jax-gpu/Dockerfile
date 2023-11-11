FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    && apt-get install --no-install-recommends -y git curl python3-pip \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax

WORKDIR /code

RUN pip install diffusers transformers

COPY test_jax.py /code/test_jax.py

CMD [ "python3", "/code/test_jax.py"]
