FROM ubuntu:21.04

ENV HOME=/workspace

WORKDIR ${HOME}

RUN useradd -M vscode && \
    chmod -R 777 ${HOME}

# Install GNU libraries and fprettify
RUN apt-get update && \
    apt-get install -y less git wget curl zsh nano build-essential openmpi-bin libopenmpi-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Compile ISPACK 1.0.4
RUN cd /usr/local/src/ && \
    wget https://www.gfd-dennou.org/arch/ispack/ispack-1.0.4.tar.gz && \
    tar xzvf ispack-1.0.4.tar.gz && \
    rm -f ispack-1.0.4.tar.gz

COPY Mkinclude /usr/local/src/ispack-1.0.4/

RUN cd /usr/local/src/ispack-1.0.4 && \
    make && \
    make clean
