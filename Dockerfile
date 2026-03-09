ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace/Chatterblez_FINITIO

COPY . /workspace/Chatterblez_FINITIO

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages .venv-qwen \
    && . .venv-qwen/bin/activate \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements_qwen.txt

RUN chmod +x start_worker.sh

EXPOSE 7860

CMD ["/workspace/Chatterblez_FINITIO/start_worker.sh"]
