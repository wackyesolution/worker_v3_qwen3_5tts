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
        pkg-config \
        libssl-dev \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN chmod +x install_azzurra.sh \
    && SKIP_FFMPEG=1 SKIP_TORCH=1 USE_SYSTEM_SITE_PACKAGES=1 ./install_azzurra.sh

RUN chmod +x start_worker.sh

EXPOSE 7860

CMD ["/workspace/Chatterblez_FINITIO/start_worker.sh"]
