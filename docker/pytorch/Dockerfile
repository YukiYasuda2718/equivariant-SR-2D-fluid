FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

ENV HOME=/workspace

EXPOSE 6006 8888

WORKDIR ${HOME}

RUN useradd -M vscode && \
    chmod -R 777 ${HOME}

# libgl1-mesa-dev is necessary for opencv2
RUN apt-get update && \
    apt-get install -y less git curl zsh libgl1-mesa-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN chsh -s /usr/bin/zsh

COPY Pipfile Pipfile.lock ${HOME}/

# Python libraries such as black are installed in ${HOME}/.local/bin 
RUN pip install -U pip pipenv && \
    export PATH="$PATH:${HOME}/.local/bin" && \
    pipenv install --system --dev --ignore-pipfile && \
    rm -rf ${HOME}/.cache/pip ${HOME}/.cache/pipenv

# Copy JupyterLab formatter configs
COPY .jupyter/ ${HOME}/.jupyter/
