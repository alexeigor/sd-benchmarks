FROM ubuntu:latest

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y git curl python3-pip && \
    pip install --upgrade pip
    
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax

WORKDIR /code

RUN pip install diffusers transformers

COPY test_jax.py /code/test_jax.py

CMD [ "python3", "/code/test_jax.py"]
