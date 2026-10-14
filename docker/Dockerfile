# fLLM runtime image: one static-feature Rust binary. The CPU kernels select AVX2 / AVX-VNNI / NEON
# at run time, so a single image serves every machine.
#   docker build -f docker/Dockerfile -t fllm .
# Limit build parallelism on shared hosts:  --build-arg JOBS=3

ARG DEBIAN_IMAGE=debian:bookworm-slim
ARG RUST_IMAGE=rust:1-slim-bookworm

# ---------------------------------------------------------------- Rust engine
FROM ${RUST_IMAGE} AS builder
ARG JOBS=4
ARG RUST_FEATURES=rust-gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config cmake clang build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build/fLLM
COPY Cargo.toml Cargo.lock ./
COPY rust ./rust
RUN CARGO_BUILD_JOBS=${JOBS} cargo build --release --locked --features "${RUST_FEATURES}" -p fllm-engine --bin fllm-rs

# ---------------------------------------------------------------- runtime
FROM ${DEBIAN_IMAGE} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/fLLM/target/release/fllm-rs /usr/local/bin/fllm-rs
RUN useradd -m fllm
USER fllm
WORKDIR /home/fllm
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://127.0.0.1:${FLLM_PORT:-8000}/health || exit 1
ENTRYPOINT ["fllm-rs"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
