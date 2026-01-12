ARG PYTORCH_TAG=2.6.0-cuda12.4-cudnn9-devel
FROM pytorch/pytorch:${PYTORCH_TAG}

# Add system dependencies
RUN apt-get update \
    # Update image
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        software-properties-common \
        curl \
        gnupg \
    && echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu jammy main" > /etc/apt/sources.list.d/ubuntu-toolchain-r-test.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F \
    && apt-get update \
    # Install essential dependencies
    && apt-get install --no-install-recommends -y \
        build-essential \
        git \
        wget \
        libxrender1 \
        libxtst6 \
        libxext6 \
        libxi6 \
        gcc-11 \
        g++-11 \
    # Install Git LFS
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install --no-install-recommends -y git-lfs \
    && git lfs install \
    # Configure gcc/g++ versions
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    # Clean up dependencies
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add ~/.local/bin to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Set work directory
WORKDIR /app/lemat-genbench

# Clone and install the package + requirements
ARG GIT_TAG=main
RUN git clone https://github.com/amorehead/lemat-genbench . --branch ${GIT_TAG} \
    && uv sync
