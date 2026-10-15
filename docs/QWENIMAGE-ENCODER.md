# Qwen-Image-2.1 VAE encoder

`c/qwenimage_encoder.h` provides the CPU f32, single-frame encoder primitive.
It shares the decoder's convolution, residual block, RMS normalization,
attention, and GEMM kernels. It does **not** enable image editing in the CLI or
server: the vision tower, reference token assembly, and DiT conditioning are
separate work. This is not a speed benchmark or an end-to-end editing claim.

## Interface

```c
QiVaeEncoder *v = qie_load("path/to/vae");
/* Check v, then allocate height/scale * width/scale * z_dim floats. */
int rc = qie_encode(v, pixels, height, width, channels, normalized_latents);
qie_free(v);
```

- `qie_load` reads `config.json`, `encoder.*`, and `quant_conv.*` from
  safetensors. It does not load decoder weights. Single-frame temporal
  downsampler convolutions are not executed and do not need weights.
- Pixels are tightly packed HWC RGB or RGBA bytes. RGB input receives opaque
  alpha for an RGBA encoder; RGBA alpha is preserved, not composited. An RGB
  model accepts RGB only. `qie_encode_f32` accepts normalized `[-1, 1]` HWC
  floats with exactly the model's configured input channels.
- Output is HWC float, with `z_dim` channels at each latent position. It is the
  deterministic posterior mode after `quant_conv`, normalized as
  `(mode - latents_mean) / latents_std`, ready for reference latent packing.
  The encoder does not sample the posterior or pack DiT patches.
- Dimensions must be positive multiples of the configured spatial scale
  (16 for Qwen-Image-2.1). Invalid dimensions/channels return `-1` before
  reading pixels. There is no implicit resize, crop, or aspect-ratio change.
- The supported architecture is residual, non-patchified Qwen-Image-2.1.
  Unsupported configurations and missing required tensors fail explicitly.
  There is no multi-frame cache or tiled encoder in this primitive.

The residual downsampler follows the pinned official implementation: spatial
padding is only bottom/right before stride-2 convolution; the AvgDown3D shortcut
includes a zero-padded temporal slot in its divisor. Both matter for single
images. The reference is
[Diffusers AutoencoderKLQwenImage21](https://github.com/huggingface/diffusers/blob/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py).

## Reproducible CPU checks

Dependency-free primitive and invalid-input tests run in `make -C c check`.
For a quick isolated run:

```sh
make -C c tests/test_qwenimage_encoder
./c/tests/test_qwenimage_encoder
```

The official oracle is opt-in; it creates random tiny weights locally, never
downloads model weights, uses two CPU threads, and removes its temporary
fixtures on both success and failure. It requires the existing pinned oracle
dependencies, including the exact Diffusers commit above:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r c/tools/requirements-qwenimage-tiny.txt
make -C c qwenimage-encoder-check PYTHON=../.venv/bin/python
# Same checks, with ASan/UBSan on the encoder, plus a saved JSON report:
.venv/bin/python c/tools/check_qwenimage_encoder.py --sanitizers --report encoder-check.json
```

The runner bounds each subprocess, reports commands/status/output/timing as
JSON, and exits nonzero on failure. It checks four sizes (16×16, 32×48, 48×32,
80×112), RGB/RGBA including identical RGB with alpha 0 and 255, normalized and
raw posterior mode, default and minimal work bands, an encoder-only checkpoint,
bad configurations and a missing intermediate weight. The existing decoder
oracle is rerun against the shared kernels. The encoder absolute tolerance is
`1e-4`; the decoder retains its existing acceptance thresholds.

The existing `qwenimage-tiny-check` CI job runs both commands (normal and
ASan/UBSan) with the pinned reference dependencies. Ordinary `make check`
still runs only dependency-free tests; it does not install Torch or Diffusers.
The sanitizer runner disables LeakSanitizer and does not claim leak coverage.

Tiny oracle agreement validates the implementation, not trained-weight visual
quality or editing instruction-following. Full trained checkpoints, non-CPU
backends, large-image peak memory, and downstream reference conditioning need
their own later validation.
