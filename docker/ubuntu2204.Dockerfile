# Dockerfile for registry.freedesktop.org/mateosss/basalt and generating .deb package

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# TODO: Add --no-install-recommends to reduce image size (requires going step-by-step)
RUN apt update && \
  apt upgrade -y && \
  apt install -y \
    gcc \
    g++ \
    cmake \
    mold \
    git \
    clang-format-15 \
    unzip \
    python3-pip \
    libtbb-dev \
    libeigen3-dev \
    libglew-dev \
    ccache \
    libgl1-mesa-dev \
    libjpeg-dev \
    libpng-dev \
    liblz4-dev \
    libbz2-dev \
    libboost-regex-dev \
    libboost-filesystem-dev \
    libboost-date-time-dev \
    libboost-program-options-dev \
    libgtest-dev \
    libopencv-dev \
    libfmt-dev \
    libyaml-cpp-dev \
    libsqlite3-dev \
    ninja-build \
    libepoxy-dev \
    jq \
    tree \
    zip \
    7zip \
    xz-utils \
    curl && \
  export CLANGD_URL=$(curl -s https://api.github.com/repos/clangd/clangd/releases/latest | jq -r '[.assets[] | select(.name | test("^clangd-linux-.*\\.zip$"))][0].browser_download_url') && \
    echo "Setting up clangd-tidy" && \
    echo "CLANGD_URL=" $CLANGD_URL && \
    curl -L --silent --show-error --fail $CLANGD_URL -o clangd-linux.zip && \
    unzip -q clangd-linux.zip && \
    rm clangd-linux.zip && \
    mv clangd* /clangd && \
    pip install clangd-tidy && \
  rm -rf /var/lib/apt/lists/* && \
  apt autoremove -y
