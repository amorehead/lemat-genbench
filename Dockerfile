ARG PYTORCH_TAG=2.6.0-cuda12.4-cudnn9-devel
FROM pytorch/pytorch:${PYTORCH_TAG}
COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/

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

# Set work directory
WORKDIR /app/lemat-genbench

# Clone the repository
ARG GIT_TAG=main
RUN git clone https://github.com/amorehead/lemat-genbench . --branch ${GIT_TAG}

# Disable development dependencies
ENV UV_NO_DEV=1

# Sync the project into a new environment, asserting the lockfile is up to date
RUN uv sync --locked
