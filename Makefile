PORT ?= 8000
THREADS ?= 8
CTX ?= 4096
MAX_SEQS ?= 16
MAX_BATCHED ?= 512
RUST_FEATURES ?= rust-gpu
FLLM_RS ?= ./run.sh
JOBS ?= 2
CUDA_ARCHS ?= 61;70;75;80;86;89;90;120
REGISTRY ?= ghcr.io/marco-lu2508
IMAGE ?= $(REGISTRY)/fllm
TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo local)

.PHONY: help install build-rust build-examples venv test test-arch test-generation test-model test-encoder docs-check lint bench serve chat build-image build-image-cuda push-image push-image-cuda run-image clean
help:
	@echo "install / build-rust / test / lint / serve MODEL=... / chat / bench / build-image"
	@echo "proof scripts: venv / build-examples / test-arch [ARCHS=...] / test-generation / test-model MODEL=x.gguf / test-encoder MODEL=x.gguf / docs-check"
install:
	./install.sh
build-rust:
	cargo build --release --locked -p fllm-engine --features "$(RUST_FEATURES)" --bin fllm-rs
test:
	cargo test --workspace --locked --features "$(RUST_FEATURES)"
lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --locked --features "$(RUST_FEATURES)" --all-targets -- -D warnings
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
FLLM_BIN ?= target/release/fllm-rs
EXAMPLES ?= $(dir $(FLLM_BIN))examples
ARCHS ?= llama qwen2 gemma2 qwen3 mixtral qwen3_moe
build-examples:
	cargo build --release --locked -p fllm-engine --features "$(RUST_FEATURES)" --bin fllm-rs --example verify_checkpoint --example tokenize --example verify_quant
venv:
	python3 -m venv .venv
	.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r scripts/requirements.txt
test-arch:
	$(PYTHON) scripts/verify_architectures.py --fllm $(FLLM_BIN) --example $(EXAMPLES)/verify_checkpoint $(ARCHS)
test-generation:
	$(PYTHON) scripts/verify_generation.py --fllm $(FLLM_BIN) $(ARCHS)
test-model:
	@test -n "$(MODEL)" || { echo "usage: make test-model MODEL=path/to/small-instruct-model.gguf [ARGS=...]"; exit 2; }
	$(PYTHON) scripts/verify_features.py --binary $(FLLM_BIN) --model "$(MODEL)" $(ARGS)
test-encoder:
	@test -n "$(MODEL)" || { echo "usage: make test-encoder MODEL=path/to/encoder.gguf [ARGS=...]"; exit 2; }
	$(PYTHON) scripts/verify_encoder.py --binary $(FLLM_BIN) --model "$(MODEL)" $(ARGS)
docs-check:
	python3 scripts/check_docs.py
serve:
	$(FLLM_RS) serve --model "$(MODEL)" --port $(PORT) --threads $(THREADS) --ctx $(CTX) --max-seqs $(MAX_SEQS) --max-batched-tokens $(MAX_BATCHED) $(ARGS)
chat:
	$(FLLM_RS) chat --url http://127.0.0.1:$(PORT) $(ARGS)
bench:
	$(FLLM_RS) bench --url http://127.0.0.1:$(PORT) $(ARGS)
build-image:
	docker build -f docker/Dockerfile --build-arg JOBS=$(JOBS) -t $(IMAGE):$(TAG) -t $(IMAGE):latest .
build-image-cuda:
	docker build -f docker/Dockerfile.cuda --build-arg JOBS=$(JOBS) --build-arg 'CUDA_ARCHS=$(CUDA_ARCHS)' -t $(IMAGE):cuda-$(TAG) -t $(IMAGE):cuda .
push-image:
	docker push $(IMAGE):$(TAG)
	docker push $(IMAGE):latest
push-image-cuda:
	docker push $(IMAGE):cuda-$(TAG)
	docker push $(IMAGE):cuda
run-image:
	docker run --rm -p 8000:8000 -v $(HOME)/models:/models:ro $(IMAGE):latest serve --host 0.0.0.0 --port 8000 --model /models/$(MODEL)
clean:
	cargo clean
