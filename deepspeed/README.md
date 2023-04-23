docker build . -t test_deepspeed

docker run -it --gpus all test_deepspeed
