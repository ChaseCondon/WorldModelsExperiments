FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && \ 
apt-get install -y python-numpy \
    cmake \
    zlib1g-dev \
    libjpeg-dev \
    libboost-all-dev \
    gcc \
    libsdl2-dev \
    wget \
    unzip \
    git \
    libgtk2.0-dev \
    python-numpy &&\
git clone https://github.com/simontudo/vizdoomgym.git && \
cd vizdoomgym && git checkout docker && \
pip install -e . && \
pip install scikit-image

ENTRYPOINT ["/bin/bash"]

