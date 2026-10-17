# SPDX-License-Identifier: Apache-2.0
{ lib
, rustPlatform
, fetchFromGitHub
, pkg-config
, shaderc
, vulkan-loader
, stdenv
, apple-sdk_14
}:

rustPlatform.buildRustPackage rec {
  pname = "fllm";
  version = "0.1.2";

  src = fetchFromGitHub {
    owner = "Marco-Lu2508";
    repo = "fLLM";
    rev = "v${version}";
    hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
  };

  cargoLock = {
    lockFile = ./../../Cargo.lock;
  };

  buildAndTestSubdir = "rust/engine";
  cargoBuildFlags = [ "--bin" "fllm-rs" ];
  buildFeatures = [ "rust-gpu" ]
    ++ lib.optionals stdenv.hostPlatform.isLinux [ "vulkan" ]
    ++ lib.optionals stdenv.hostPlatform.isDarwin [ "metal" ]
    ++ lib.optionals stdenv.hostPlatform.isx86_64 [ "avx512" ]
    ++ lib.optionals stdenv.hostPlatform.isAarch64 [ "arm-simd" ];

  nativeBuildInputs = [ pkg-config ]
    ++ lib.optionals stdenv.hostPlatform.isLinux [ shaderc ];

  buildInputs = lib.optionals stdenv.hostPlatform.isLinux [ vulkan-loader ]
    ++ lib.optionals stdenv.hostPlatform.isDarwin [ apple-sdk_14 ];

  env.GLSLC = lib.optionalString stdenv.hostPlatform.isLinux "${shaderc.bin}/bin/glslc";

  doCheck = true;

  postInstall = ''
    install -Dm644 LICENSE NOTICE THIRD_PARTY.md -t "$out/share/doc/fllm"
    cp -r licenses "$out/share/doc/fllm/licenses"
  '';

  meta = with lib; {
    description = "Standalone Rust LLM and encoder inference server and CLI";
    homepage = "https://github.com/Marco-Lu2508/fLLM";
    license = licenses.asl20;
    mainProgram = "fllm-rs";
    platforms = platforms.unix ++ platforms.windows;
  };
}
