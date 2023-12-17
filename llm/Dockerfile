FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    && apt-get install --no-install-recommends -y git curl python3-pip \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /code

RUN pip install transformers torch

COPY test_llm.py /code/test_llm.py

CMD python3 /code/test_llm.py