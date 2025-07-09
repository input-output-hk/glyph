FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    xz-utils \
    qemu-user-static \
    binfmt-support \
    && ln -sf /usr/bin/qemu-riscv32-static /usr/bin/qemu-riscv32 \
    && rm -rf /var/lib/apt/lists/*

# Install Zig
RUN curl -L https://ziglang.org/download/0.14.1/zig-x86_64-linux-0.14.1.tar.xz | tar -xJ -C /opt
ENV PATH="/opt/zig-x86_64-linux-0.14.1:$PATH"

WORKDIR /work
CMD ["zig", "build", "test", "--summary", "all"]
