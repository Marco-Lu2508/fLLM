# SPDX-License-Identifier: Apache-2.0
class Fllm < Formula
  desc "Standalone Rust LLM and encoder inference server and CLI"
  homepage "https://github.com/Marco-Lu2508/fLLM"
  url "https://github.com/Marco-Lu2508/fLLM/archive/refs/tags/v0.1.2.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "Apache-2.0"
  head "https://github.com/Marco-Lu2508/fLLM.git", branch: "main"

  depends_on "rust" => :build

  def install
    features = OS.mac? ? "rust-gpu,metal" : "rust-gpu"
    system "cargo", "install", *std_cargo_args(path: "rust/engine"), "--features", features, "--bin", "fllm-rs"
    prefix.install "LICENSE", "NOTICE", "THIRD_PARTY.md"
    (prefix/"licenses").install Dir["licenses/*"]
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/fllm-rs --version")
  end
end
