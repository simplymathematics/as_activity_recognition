FROM ubuntu:22.04
RUN apt update
RUN apt -y upgrade
RUN apt install -y python3 python3-pip build-essential libssl-dev libffi-dev python3-dev
# RUN python3 -m pip install nvidia-pyindex
# RUN python3 -m pip install nvidia-cuda-runtime-cu11
# RUN pip3 install tensorflow-gpu
ADD . /project
WORKDIR /project
RUN python3 -m pip install -r requirements.txt
RUN dvc repro --force
WORKDIR /project/results
CMD ["python3", "-m", "http.server", "8000"]