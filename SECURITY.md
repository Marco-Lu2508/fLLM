# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting instead:
**Security** tab of this repository, **Report a vulnerability**. Include the version or commit, what you did and what you
expected to happen. You will get an answer as soon as possible; this is a small project, so please be patient.

## Scope and what to expect

- `fllm-rs serve` is built for use on your own machine or inside a trusted network. It listens on `127.0.0.1` by default.
- If you expose the server beyond that, set an API key (`--api-key` or `FLLM_API_KEY`) and put a reverse proxy in front of it
  (TLS, request size and time limits, rate limiting). The server does not yet enforce request timeouts of its own and compares the
  API key with a plain string comparison; both are known and will be improved.
- Model files are data, but a malformed GGUF or checkpoint can still crash the loader. Only load models you trust.
- Release assets carry a `.sha256` file next to them; `install.sh` and `install.ps1` verify it.

## Supported versions

Only the newest release and the `main` branch receive fixes.
