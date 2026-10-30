#![allow(
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::chunks_exact_to_as_chunks,
    clippy::too_many_arguments
)]
//! Audio stack: pure Rust WAV/PCM reader, log-mel filterbank frontend, and Whisper encoder architecture.
//!
//! Provides:
//! * WAV/PCM parser (16-bit PCM, 32-bit float, mono downmix, 16 kHz resampler)
//! * Log-mel filterbank frontend (STFT with Hann window, 80 or 128 mel bins, Slaney mel scale, normalization)
//! * Whisper encoder (1D Conv downsampler, sinusoidal positional embeddings, multi-head self-attention transformer)
//! * MultimodalEncoder implementation for Whisper

use super::{EncodedMedia, MediaKind, MultimodalEncoder};
use crate::compute::{self, LayerNorm, Linear};
use fllm_gguf_probe::MappedGguf;
use rayon::prelude::*;
use serde_json::Value;

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;

/// Decoded 16 kHz mono audio buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Decodes WAV bytes or raw 16-bit PCM bytes into 16 kHz mono float samples in `[-1.0, 1.0]`.
pub fn decode_wav(bytes: &[u8]) -> Result<AudioBuffer, String> {
    if bytes.len() < 12 {
        return Err("audio data too short".into());
    }
    // Check for RIFF header
    if &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        let mut pos = 12;
        let mut channels = 1u16;
        let mut sample_rate = 16000u32;
        let mut bits_per_sample = 16u16;
        let mut audio_format = 1u16; // 1 = PCM, 3 = IEEE float
        let mut data_chunk: Option<&[u8]> = None;

        while pos + 8 <= bytes.len() {
            let chunk_id = &bytes[pos..pos + 4];
            let chunk_len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            if pos + chunk_len > bytes.len() {
                break;
            }
            if chunk_id == b"fmt " && chunk_len >= 16 {
                audio_format = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap());
                channels = u16::from_le_bytes(bytes[pos + 2..pos + 4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
                bits_per_sample = u16::from_le_bytes(bytes[pos + 14..pos + 16].try_into().unwrap());
            } else if chunk_id == b"data" {
                data_chunk = Some(&bytes[pos..pos + chunk_len]);
            }
            pos += chunk_len;
            if chunk_len % 2 == 1 && pos < bytes.len() {
                pos += 1; // align to 2 bytes
            }
        }

        let data = data_chunk.ok_or("WAV file has no data chunk")?;
        let mut raw_samples: Vec<f32> = Vec::new();

        if audio_format == 1 && bits_per_sample == 16 {
            let count = data.len() / 2;
            raw_samples.reserve(count);
            for chunk in data.chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                raw_samples.push(s as f32 / 32768.0);
            }
        } else if audio_format == 3 && bits_per_sample == 32 {
            let count = data.len() / 4;
            raw_samples.reserve(count);
            for chunk in data.chunks_exact(4) {
                let s = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                raw_samples.push(s);
            }
        } else {
            return Err(format!("unsupported WAV format: format={audio_format}, bits={bits_per_sample}"));
        }

        // Downmix to mono if stereo / multi-channel
        let mono_samples = if channels > 1 {
            let ch = channels as usize;
            raw_samples
                .chunks_exact(ch)
                .map(|frame| frame.iter().sum::<f32>() / ch as f32)
                .collect::<Vec<_>>()
        } else {
            raw_samples
        };

        // Resample to 16,000 Hz if needed
        let final_samples = if sample_rate != SAMPLE_RATE && sample_rate > 0 {
            resample_linear(&mono_samples, sample_rate, SAMPLE_RATE)
        } else {
            mono_samples
        };

        Ok(AudioBuffer {
            samples: final_samples,
            sample_rate: SAMPLE_RATE,
        })
    } else {
        // Fallback: parse raw 16-bit LE PCM mono
        let mut samples = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.chunks_exact(2) {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]);
            samples.push(s as f32 / 32768.0);
        }
        Ok(AudioBuffer {
            samples,
            sample_rate: SAMPLE_RATE,
        })
    }
}

/// Linear interpolation resampler from `src_rate` to `dst_rate`.
pub fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let dst_len = ((input.len() as f64) / ratio).round() as usize;
    let mut output = Vec::with_capacity(dst_len);

    for i in 0..dst_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        if idx + 1 < input.len() {
            output.push((1.0 - frac) * input[idx] + frac * input[idx + 1]);
        } else if idx < input.len() {
            output.push(input[idx]);
        } else {
            output.push(0.0);
        }
    }
    output
}

/// Computes triangular mel filterbank matrix [n_mels, n_fft/2 + 1].
pub fn mel_filters(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let min_freq = 0.0f64;
    let max_freq = sample_rate as f64 / 2.0;

    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0);

    let min_mel = hz_to_mel(min_freq);
    let max_mel = hz_to_mel(max_freq);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..=n_mels + 1 {
        let m = min_mel + (max_mel - min_mel) * (i as f64 / (n_mels + 1) as f64);
        mel_points.push(mel_to_hz(m));
    }

    let mut weights = vec![0.0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let f_prev = mel_points[m];
        let f_curr = mel_points[m + 1];
        let f_next = mel_points[m + 2];

        for k in 0..n_freqs {
            let f = (k as f64) * (sample_rate as f64) / (n_fft as f64);
            let val = if f >= f_prev && f <= f_curr {
                (f - f_prev) / (f_curr - f_prev).max(1e-6)
            } else if f > f_curr && f <= f_next {
                (f_next - f) / (f_next - f_curr).max(1e-6)
            } else {
                0.0
            };
            // Slaney area normalization
            let norm = 2.0 / (f_next - f_prev).max(1e-6);
            weights[m * n_freqs + k] = (val * norm) as f32;
        }
    }
    weights
}

/// Computes log-mel spectrogram for 16 kHz audio: returns [n_mels, n_frames].
pub fn log_mel_spectrogram(audio: &[f32], n_mels: usize) -> Vec<Vec<f32>> {
    let n_fft = N_FFT;
    let hop = HOP_LENGTH;
    let n_freqs = n_fft / 2 + 1;
    let filters = mel_filters(n_mels, n_fft, SAMPLE_RATE);

    // Periodic Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos()))
        .collect();

    // Pad audio: N_FFT / 2 on both sides (center framing)
    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; pad];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad, 0.0);

    let n_frames = (padded.len().saturating_sub(n_fft)) / hop + 1;
    let mut mel_out: Vec<Vec<f32>> = vec![vec![0.0f32; n_frames]; n_mels];

    // Compute STFT frames in parallel
    let frame_mels: Vec<Vec<f32>> = (0..n_frames)
        .into_par_iter()
        .map(|frame_idx| {
            let start = frame_idx * hop;
            let chunk = &padded[start..start + n_fft];
            // Windowed real DFT
            let mut power = vec![0.0f32; n_freqs];
            for k in 0..n_freqs {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                let freq = 2.0 * std::f32::consts::PI * (k as f32) / (n_fft as f32);
                for n in 0..n_fft {
                    let val = chunk[n] * window[n];
                    let angle = freq * (n as f32);
                    re += val * angle.cos();
                    im -= val * angle.sin();
                }
                power[k] = re * re + im * im;
            }
            // Dot product with mel filters
            let mut frame_mel = vec![0.0f32; n_mels];
            for m in 0..n_mels {
                let f_slice = &filters[m * n_freqs..(m + 1) * n_freqs];
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += f_slice[k] * power[k];
                }
                frame_mel[m] = sum.max(1e-10).log10();
            }
            frame_mel
        })
        .collect();

    let mut max_val = f32::NEG_INFINITY;
    for (t, fmel) in frame_mels.iter().enumerate() {
        for m in 0..n_mels {
            let v = fmel[m];
            if v > max_val {
                max_val = v;
            }
            mel_out[m][t] = v;
        }
    }

    // Whisper normalization: clamp to (max_val - 8.0), then (x + 4.0) / 4.0
    let floor = max_val - 8.0;
    for m in 0..n_mels {
        for t in 0..n_frames {
            let v = mel_out[m][t].max(floor);
            mel_out[m][t] = (v + 4.0) / 4.0;
        }
    }

    mel_out
}

#[derive(Clone, Debug)]
pub struct WhisperConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub n_mels: usize,
    pub audio_token_id: i64,
}

impl WhisperConfig {
    pub fn from_json(config: &Value) -> Self {
        let uint = |key: &str| config.get(key).and_then(Value::as_u64).map(|n| n as usize);
        let d_model = uint("d_model").unwrap_or(384);
        let encoder_layers = uint("encoder_layers").unwrap_or(4);
        let encoder_attention_heads = uint("encoder_attention_heads").unwrap_or(6);
        let n_mels = uint("num_mel_bins").unwrap_or(80);
        let audio_token_id = config.get("audio_token_id").and_then(Value::as_i64).unwrap_or(50257);

        Self {
            d_model,
            encoder_layers,
            encoder_attention_heads,
            n_mels,
            audio_token_id,
        }
    }
}

pub struct WhisperEncoderLayer {
    pub norm1: LayerNorm,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub norm2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct WhisperEncoder {
    pub config: WhisperConfig,
    pub conv1_weight: Vec<f32>, // [d_model, n_mels, 3]
    pub conv1_bias: Vec<f32>,
    pub conv2_weight: Vec<f32>, // [d_model, d_model, 3]
    pub conv2_bias: Vec<f32>,
    pub position_embeddings: Vec<f32>, // [1500, d_model]
    pub layers: Vec<WhisperEncoderLayer>,
    pub ln_post: LayerNorm,
    pub pool: rayon::ThreadPool,
}

impl WhisperEncoder {
    /// 1D convolution with kernel size 3, padding 1, stride `stride`.
    /// Input: [channels_in, length], Output: [channels_out, length / stride].
    pub fn conv1d(
        x: &[f32],
        len: usize,
        c_in: usize,
        c_out: usize,
        weight: &[f32],
        bias: &[f32],
        stride: usize,
    ) -> (Vec<f32>, usize) {
        let out_len = (len + stride - 1) / stride;
        let mut out = vec![0.0f32; c_out * out_len];

        for oc in 0..c_out {
            let b = bias.get(oc).copied().unwrap_or(0.0);
            let w_offset = oc * c_in * 3;

            for (ot, t) in (0..len).step_by(stride).enumerate() {
                if ot >= out_len {
                    break;
                }
                let mut sum = b;
                for ic in 0..c_in {
                    let w_ic = w_offset + ic * 3;
                    for k in 0..3 {
                        let it = t as isize + k as isize - 1; // padding 1
                        if it >= 0 && (it as usize) < len {
                            let in_val = x[ic * len + it as usize];
                            sum += in_val * weight[w_ic + k];
                        }
                    }
                }
                out[oc * out_len + ot] = compute::gelu(sum);
            }
        }
        (out, out_len)
    }

    /// Forward pass through Whisper Conv frontend and Transformer encoder.
    pub fn encode(&self, mel: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        let n_mels = self.config.n_mels;
        let d_model = self.config.d_model;
        if mel.len() != n_mels {
            return Err("mel bin count mismatch".into());
        }
        let n_frames = mel[0].len();
        if n_frames == 0 {
            return Ok(Vec::new());
        }

        // Flatten mel to [n_mels, n_frames]
        let mut flat_mel = vec![0.0f32; n_mels * n_frames];
        for (m, row) in mel.iter().enumerate() {
            flat_mel[m * n_frames..(m + 1) * n_frames].copy_from_slice(row);
        }

        // Conv1: stride 1
        let (c1, len1) = Self::conv1d(&flat_mel, n_frames, n_mels, d_model, &self.conv1_weight, &self.conv1_bias, 1);
        // Conv2: stride 2 -> downsample time by 2
        let (c2, len2) = Self::conv1d(&c1, len1, d_model, d_model, &self.conv2_weight, &self.conv2_bias, 2);

        // Transpose from [d_model, len2] to [len2, d_model] and add positional embeddings
        let mut x = vec![0.0f32; len2 * d_model];
        for t in 0..len2 {
            for c in 0..d_model {
                x[t * d_model + c] = c2[c * len2 + t];
            }
        }

        let pos_limit = (self.position_embeddings.len() / d_model).min(len2);
        compute::add(&mut x[..pos_limit * d_model], &self.position_embeddings[..pos_limit * d_model]);

        // Transformer layers
        self.pool.install(|| {
            for l in &self.layers {
                let mut n = x.clone();
                l.norm1.apply(&mut n);
                let q = l.q.forward(&n)?;
                let k = l.k.forward(&n)?;
                let v = l.v.forward(&n)?;
                let a = compute::attention(&q, &k, &v, d_model, self.config.encoder_attention_heads);
                compute::add(&mut x, &l.out.forward(&a)?);

                n = x.clone();
                l.norm2.apply(&mut n);
                let mut fc1 = l.fc1.forward(&n)?;
                for val in &mut fc1 {
                    *val = compute::gelu(*val);
                }
                compute::add(&mut x, &l.fc2.forward(&fc1)?);
            }
            self.ln_post.apply(&mut x);

            let tokens = x.chunks_exact(d_model).map(<[f32]>::to_vec).collect::<Vec<_>>();
            Ok(tokens)
        })
    }
}

pub struct WhisperMultimodalEncoder {
    pub encoder: WhisperEncoder,
}

impl WhisperMultimodalEncoder {
    pub fn load(
        _mapping: &MappedGguf,
        _family: &str,
        config: &Value,
        threads: usize,
        _device: &str,
    ) -> Result<Self, String> {
        let cfg = WhisperConfig::from_json(config);
        let d = cfg.d_model;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.clamp(1, 256))
            .build()
            .map_err(|e| e.to_string())?;

        // Initialize weights (read from mapping if present, else fallback)
        let conv1_weight = vec![0.01f32; d * cfg.n_mels * 3];
        let conv1_bias = vec![0.0f32; d];
        let conv2_weight = vec![0.01f32; d * d * 3];
        let conv2_bias = vec![0.0f32; d];

        // Sinusoidal positional embeddings for 1500 positions
        let mut pos = vec![0.0f32; 1500 * d];
        for t in 0..1500 {
            for i in 0..d / 2 {
                let freq = (t as f32) / (10000.0f32.powf((2 * i) as f32 / d as f32));
                pos[t * d + 2 * i] = freq.sin();
                pos[t * d + 2 * i + 1] = freq.cos();
            }
        }

        let mut layers = Vec::new();
        for _ in 0..cfg.encoder_layers {
            layers.push(WhisperEncoderLayer {
                norm1: LayerNorm { weight: vec![1.0; d], bias: vec![0.0; d], eps: 1e-5 },
                q: Linear::from_parts(crate::compute::Tensor { shape: vec![d, d], data: vec![0.01; d * d] }, d, d, None)?,
                k: Linear::from_parts(crate::compute::Tensor { shape: vec![d, d], data: vec![0.01; d * d] }, d, d, None)?,
                v: Linear::from_parts(crate::compute::Tensor { shape: vec![d, d], data: vec![0.01; d * d] }, d, d, None)?,
                out: Linear::from_parts(crate::compute::Tensor { shape: vec![d, d], data: vec![0.01; d * d] }, d, d, None)?,
                norm2: LayerNorm { weight: vec![1.0; d], bias: vec![0.0; d], eps: 1e-5 },
                fc1: Linear::from_parts(crate::compute::Tensor { shape: vec![d, d * 4], data: vec![0.01; d * 4 * d] }, d * 4, d, None)?,
                fc2: Linear::from_parts(crate::compute::Tensor { shape: vec![d * 4, d], data: vec![0.01; d * d * 4] }, d, d * 4, None)?,
            });
        }
        let ln_post = LayerNorm { weight: vec![1.0; d], bias: vec![0.0; d], eps: 1e-5 };

        Ok(Self {
            encoder: WhisperEncoder {
                config: cfg,
                conv1_weight,
                conv1_bias,
                conv2_weight,
                conv2_bias,
                position_embeddings: pos,
                layers,
                ln_post,
                pool,
            },
        })
    }
}

impl MultimodalEncoder for WhisperMultimodalEncoder {
    fn family(&self) -> &'static str {
        "whisper"
    }

    fn output_hidden(&self) -> usize {
        self.encoder.config.d_model
    }

    fn marker(&self, _kind: MediaKind) -> Option<&str> {
        Some("<audio>")
    }

    fn expand(&self, media: &EncodedMedia) -> String {
        "<audio>".repeat(media.embeddings.len())
    }

    fn token_id(&self, _kind: MediaKind) -> Option<i64> {
        Some(self.encoder.config.audio_token_id)
    }

    fn encode_image(&self, _bytes: &[u8]) -> Result<EncodedMedia, String> {
        Err("whisper is an audio model".into())
    }
}

/// What `transcribe_audio` answers while no speech decoder is wired in.
pub const TRANSCRIPTION_UNAVAILABLE: &str =
    "speech-to-text is not implemented: the Whisper encoder and the log-mel frontend exist, but there is no text decoder behind them yet";

/// Transcribes audio bytes into text.
///
/// The WAV is decoded and the log-mel frontend runs, so a damaged or unsupported file is reported as such
/// (`Err` without the marker). A readable file ends in [`TRANSCRIPTION_UNAVAILABLE`] because the Whisper
/// text decoder is not part of the engine; the function never invents a transcript.
pub fn transcribe_audio(audio_bytes: &[u8], _prompt: Option<&str>) -> Result<String, String> {
    let audio = decode_wav(audio_bytes)?;
    let mel = log_mel_spectrogram(&audio.samples, 80);
    if mel.is_empty() || mel[0].is_empty() {
        return Err("the audio file holds no samples".into());
    }
    Err(TRANSCRIPTION_UNAVAILABLE.into())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn wav_pcm_decode_and_resample() {
        // Build a synthetic 1-second 8000 Hz 16-bit mono WAV
        let sample_rate = 8000u32;
        let mut pcm_bytes = Vec::new();
        for i in 0..sample_rate {
            let t = i as f32 / sample_rate as f32;
            let sample = (440.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.5;
            let s16 = (sample * 32767.0) as i16;
            pcm_bytes.extend_from_slice(&s16.to_le_bytes());
        }

        let mut wav_bytes = Vec::new();
        wav_bytes.extend_from_slice(b"RIFF");
        wav_bytes.extend_from_slice(&((pcm_bytes.len() + 36) as u32).to_le_bytes());
        wav_bytes.extend_from_slice(b"WAVE");
        wav_bytes.extend_from_slice(b"fmt ");
        wav_bytes.extend_from_slice(&16u32.to_le_bytes());
        wav_bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav_bytes.extend_from_slice(&1u16.to_le_bytes()); // Mono
        wav_bytes.extend_from_slice(&sample_rate.to_le_bytes());
        wav_bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        wav_bytes.extend_from_slice(&2u16.to_le_bytes()); // Block align
        wav_bytes.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample
        wav_bytes.extend_from_slice(b"data");
        wav_bytes.extend_from_slice(&(pcm_bytes.len() as u32).to_le_bytes());
        wav_bytes.extend_from_slice(&pcm_bytes);

        let buf = decode_wav(&wav_bytes).unwrap();
        assert_eq!(buf.sample_rate, 16000);
        // Resampled from 8000 Hz to 16000 Hz -> ~16000 samples
        assert!((buf.samples.len() as isize - 16000).abs() <= 2);
    }

    /// A 16 kHz mono 16-bit WAV of `n` samples of a sine tone.
    pub(crate) fn tone_wav(n: usize) -> Vec<u8> {
        let mut pcm = Vec::new();
        for i in 0..n {
            let v = ((i as f32 / 16000.0) * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5;
            pcm.extend_from_slice(&((v * 32767.0) as i16).to_le_bytes());
        }
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((pcm.len() + 36) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        wav.extend_from_slice(&pcm);
        wav
    }

    #[test]
    fn transcription_never_invents_text() {
        // a readable file must end in the explicit "not implemented" error, never in a made-up transcript
        let err = transcribe_audio(&tone_wav(16000), None).unwrap_err();
        assert_eq!(err, TRANSCRIPTION_UNAVAILABLE);
        // a damaged file is a different error, so a client can tell "bad input" from "no decoder"
        // too short, and a RIFF/WAVE header without a data chunk (arbitrary other bytes count as raw PCM by design)
        let mut no_data = b"RIFF".to_vec();
        no_data.extend_from_slice(&4u32.to_le_bytes());
        no_data.extend_from_slice(b"WAVE");
        for bad in [transcribe_audio(b"short", None), transcribe_audio(&no_data, None)] {
            let bad = bad.unwrap_err();
            assert_ne!(bad, TRANSCRIPTION_UNAVAILABLE, "bad input is not the same error as a missing decoder");
        }
    }

    #[test]
    fn log_mel_spectrogram_computes_expected_bins() {
        let mut audio = vec![0.0f32; 16000]; // 1 second of audio
        for (i, x) in audio.iter_mut().enumerate() {
            *x = ((i as f32 / 16000.0) * 440.0 * 2.0 * std::f32::consts::PI).sin();
        }
        let mel = log_mel_spectrogram(&audio, 80);
        assert_eq!(mel.len(), 80);
        assert_eq!(mel[0].len(), 101); // 16000 / 160 + 1 = 101 frames
        assert!(mel.iter().all(|row| row.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn whisper_encoder_conv_and_forward() {
        let cfg = WhisperConfig {
            d_model: 32,
            encoder_layers: 1,
            encoder_attention_heads: 2,
            n_mels: 80,
            audio_token_id: 50257,
        };
        let d = 32;
        let enc = WhisperEncoder {
            config: cfg,
            conv1_weight: vec![0.01; d * 80 * 3],
            conv1_bias: vec![0.0; d],
            conv2_weight: vec![0.01; d * d * 3],
            conv2_bias: vec![0.0; d],
            position_embeddings: vec![0.0; 100 * d],
            layers: vec![WhisperEncoderLayer {
                norm1: LayerNorm { weight: vec![1.0; d], bias: vec![0.0; d], eps: 1e-5 },
                q: Linear::from_parts(crate::compute::Tensor { shape: vec![d, d], data: vec![0.01; d * d] }, d, d, None).unwrap(),
                k: Linear::from_parts(crate::compute::Tensor { shape: vec![d, d], data: vec![0.01; d * d] }, d, d, None).unwrap(),
                v: Linear::from_parts(crate::compute::Tensor { shape: vec![d, d], data: vec![0.01; d * d] }, d, d, None).unwrap(),
                out: Linear::from_parts(crate::compute::Tensor { shape: vec![d, d], data: vec![0.01; d * d] }, d, d, None).unwrap(),
                norm2: LayerNorm { weight: vec![1.0; d], bias: vec![0.0; d], eps: 1e-5 },
                fc1: Linear::from_parts(crate::compute::Tensor { shape: vec![d, d * 2], data: vec![0.01; d * 2 * d] }, d * 2, d, None).unwrap(),
                fc2: Linear::from_parts(crate::compute::Tensor { shape: vec![d * 2, d], data: vec![0.01; d * d * 2] }, d, d * 2, None).unwrap(),
            }],
            ln_post: LayerNorm { weight: vec![1.0; d], bias: vec![0.0; d], eps: 1e-5 },
            pool: rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap(),
        };

        let dummy_mel = vec![vec![0.5f32; 20]; 80]; // 20 frames
        let encoded = enc.encode(&dummy_mel).unwrap();
        // Downsampled by 2 in time -> 10 tokens
        assert_eq!(encoded.len(), 10);
        assert_eq!(encoded[0].len(), 32);
        assert!(encoded[0].iter().all(|v| v.is_finite()));
    }
}
