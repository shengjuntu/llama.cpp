# Qwen3 ASR and forced alignment details

Status: approved by the contributor and implemented locally. Real CPU inference and reference comparisons pass. See [usage and validation](../qwen3-asr-details.md) for commands, results, and remaining limits.

Target: the `shengjuntu/llama.cpp` fork, inspected at `5d806aa2575e01e126651fd69ab1ab6cefff861d`.

## Goal and first implementation boundary

Implement one native GGML service that accepts audio, runs Qwen3-ASR, aligns its final transcript with Qwen3-ForcedAligner, and returns the sample-based details consumed by the existing audio.cpp client.

The deployed configuration described in audio.cpp's optimization report uses `return_timestamps=true`, `audio_chunk_mode=none`, and no Silero VAD. Match this path first. Long-audio VAD, overlap handling, and chunk aggregation are a subsequent extension, not prerequisites for reproducing that deployment. Reject inputs exceeding the supported duration or either model's context capacity; never silently truncate audio or alignment text.

The acceptance target is `POST /v1/audio/transcriptions/details` in `llama-server`. A standalone alignment CLI can help validation, but is not the feature's completion criterion.

Use native GGML and the existing backend selection, including MUSA. Python is permitted for conversion and reference comparison, not required by the running service. This work introduces no custom decode buckets. Benchmark existing llama.cpp graph and KV behavior before considering separate decode optimizations.

## Request and response contract

The first compatibility target is multipart WAV upload with `file`, `model`, optional `language`, and optional recognition `prompt`. The prompt supplies recognition context; it is never the transcript passed to the aligner. JSON audio inputs supported by audio.cpp need a separate compatibility inventory before claiming complete API equivalence.

The details route runs alignment explicitly. In audio.cpp the route itself only retains detail fields: timestamp production is enabled by request or session defaults. Make the new route's behavior unambiguous and fail if the companion model is unavailable.

| Field | Meaning |
| --- | --- |
| `text` | Final ASR transcript, with punctuation retained after removing model protocol markers. |
| `language` | Parsed recognition language, when available. |
| `words[].word` | Alignment unit; Chinese characters and English words are initial parity cases. |
| `words[].start_sample` | Start frame on the original input audio timeline. |
| `words[].end_sample` | End frame on the same timeline. |
| `words[].confidence` | `0.0` as audio.cpp's existing unavailable-confidence sentinel; not a calibrated probability. |
| `sample_rate` | Original input sample rate used for all offsets. |
| `timing.wall_ms` | Combined ASR and alignment execution time, with scope documented to match the reference measurement. |
| `timing.audio_duration_ms` | Duration computed from original frames and sample rate. |
| `timing.rtf` | `wall_ms / audio_duration_ms`, with zero-duration input handled explicitly. |

Sample offsets count frames per channel, not interleaved scalar samples. Preserve original rate and frame count before mono conversion and resampling to the model rate. Use integer arithmetic when converting timestamp classes to sample offsets.

Retain audio.cpp's omission of empty optional detail arrays. Do not synthesize speaker turns or sentence segments from word timestamps. Preserve ASR punctuation in `text`; alignment normalization must not replace the transcript with space-joined units.

Reject `stream=true` on details with HTTP 400, matching audio.cpp. Keep the existing transcription streaming path separate from offline alignment. An empty recognized transcript returns an empty result without running alignment. Unsupported alignment languages, incomplete ASR generation, and malformed timestamp output must produce explicit errors rather than successful responses with guessed timestamps.

## Pipeline and ownership

1. Decode the uploaded audio and retain its original timeline metadata. Validate the input and both model capacity limits before expensive inference where possible.
2. Build the official Qwen-ASR prompt and submit recognition through the existing server task queue. Parse the final language and transcript from the model's output protocol, including the ASR text separator.
3. Normalize alignment units from that final transcript. Keep the displayed transcript separately.
4. Encode the audio with the aligner's own audio encoder. Build its audio-plus-text prompt with two timestamp markers per unit and no ASR chat wrapper.
5. Evaluate the causal decoder over the supplied sequence, collecting timestamp-class outputs. This is a non-generative pass, potentially divided into microbatches while retaining KV state; it does not sample timestamp tokens autoregressively.
6. Apply timestamp postprocessing, convert to the original timeline, and serialize details after both stages finish.

Keep two independent model and context pairs resident: ASR decoder/projector and aligner decoder/projector. Reuse encoder graph code and compatible preprocessing, but never reuse learned ASR audio embeddings as aligner embeddings. The encoder weights differ.

Attach a persistent alignment worker to the server lifecycle. It owns the aligner context and serializes its inference jobs. HTTP workers may wait for recognition and alignment; the ASR inference thread must not execute alignment or block while waiting for it. Queue admission, deadline/cancellation, shutdown, and model reload must cover both stages. Start with single-model inference mode and an optional companion aligner; audit router and sleep modes explicitly before claiming they work with the feature.

Proposed configuration consists of an aligner decoder path, aligner projector path, and independent context/backend settings. Model paths come from server startup configuration, not arbitrary request fields. The service should expose a generic optional alignment stage rather than putting Qwen-specific tensor operations into HTTP handlers.

## Model implementation

Use the official Qwen3-ForcedAligner-0.6B checkpoint as the reference. The proposed backbone is the existing Qwen3 decoder implementation, with the existing `qwen3a` encoder graph reused by configuration.

The aligner replaces language-model generation with a 5000-class timestamp head. Store the class count, timestamp marker identity, and time step from model configuration in GGUF metadata. The published checkpoint uses an 80 ms time step. Validate these values when loading; do not infer time resolution from vocabulary size.

Define a distinct `qwen3aligner` architecture so a timestamp classifier cannot be mistaken for a normal ASR generation model. Keep causal attention and normal Qwen3 Q/K normalization. Original checkpoints describe interleaved M-RoPE, while the native HF token-classification layout describes ordinary Qwen3 RoPE. For audio/text with identical position coordinates, verify the proposed one-dimensional RoPE conversion numerically against the selected reference before accepting it.

The current logits buffer is allocated and copied with vocabulary-sized rows. Substituting a 5000-column head into that path is invalid. The proposed bounded implementation exports classifier scores through unpooled per-token outputs with output width 5000 and no generation logits. It requires a scoped output-allocation change to avoid an unused vocabulary-sized buffer. Sampling and pooled reranking are not alignment output interfaces.

Embedding mode currently requests every token's output. A timestamp-only `batch.logits` mask cannot be assumed to reduce output rows. Read each microbatch's timestamp rows before the next decode, keep output indexing explicit, and bound the working buffer. A later selective per-token classification API would require its own design review.

Text normalization and timestamp repair must be checked against both the official reference and the deployed audio.cpp behavior. Official monotonic repair and audio.cpp's zero-duration handling are not automatically identical. Preserve an audio.cpp-compatible postprocessing path for deployment comparison, and document any deliberate differences. Do not equate raw timestamp scores with recognition confidence. Japanese and Korean require dedicated segmentation parity work before advertising support for all aligner languages.

## Source touch points

| Area | Existing files and proposed additions |
| --- | --- |
| Conversion | `conversion/qwen3vl.py`, `conversion/__init__.py`, and GGUF architecture/metadata/tensor mappings. Detect the original aligner config without misclassifying ordinary ASR checkpoints. |
| Decoder | `src/llama-arch.h`, `src/llama-arch.cpp`, `src/llama-model.cpp`, model declarations/build registration, and a dedicated `src/models/qwen3aligner.cpp`. |
| Outputs | A scoped change in `src/llama-context.cpp` for non-generative classification output allocation and validation. |
| Audio | Reuse `tools/mtmd/models/qwen3a.cpp`; inspect `mtmd-audio.cpp` and `mtmd.cpp` for frontend, valid-frame, and prompt parity. |
| Alignment driver | A reusable internal alignment helper owning model loading, prompt assembly, microbatch output extraction, normalization, and timestamp postprocessing. |
| Service | `tools/server/server-context.*`, `server-chat.*`, `server.cpp`, and a small alignment worker component with build registration. |
| Configuration | Existing common parameter parsing and server model lifecycle; companion settings must not leak into unrelated models. |
| Documentation | Conversion, launch, API compatibility, supported languages, limits, and validation results. Options and commands become usage documentation only after implementation. |

Audio.cpp-native GGUF files are not assumed compatible with llama.cpp. Convert from the original checkpoint into llama.cpp's decoder plus projector layout.

## Validation and completion

1. Establish reference ASR prompt tokens, audio features, transcript, aligner prompt tokens, and timestamp-class outputs for the same recordings. Verify existing Qwen-ASR behavior before attributing differences to the new classifier.
2. Compare float GGUF outputs against the official implementation and deployed audio.cpp output. Diagnose raw model differences separately from normalization and timestamp repair differences. Zero drift is an acceptance goal, not a result established by this design.
3. Exercise Chinese, English, mixed text, punctuation, empty speech, non-16 kHz input, invalid files, context overflow, and missing companion configuration. Verify original-rate sample offsets and the details error contract.
4. Exercise repeated requests, simultaneous requests, cancellation between stages, and shutdown. Check context isolation and that alignment cannot stall the ASR inference loop.
5. Reuse existing test infrastructure. Do not create new files under `tests/` without the approval required by this repository.
6. Build and run on CPU first. Then compare warmed combined RTF, stage times, peak memory, and timestamp output on the user's MUSA SDK 5.1.0 system. CPU compilation does not establish MUSA runtime correctness or speed.

The feature is complete only when an HTTP request runs both real models and returns validated details. A compiling converter, a standalone aligner, or a mocked endpoint alone is insufficient.

## Reference sources

- [Qwen3-ASR official implementation](https://github.com/QwenLM/Qwen3-ASR)
- [Official forced-alignment driver](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py)
- [Native HF aligner checkpoint](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
- [Target llama.cpp fork](https://github.com/shengjuntu/llama.cpp)
- [audio.cpp reference fork](https://github.com/shengjuntu/audio.cpp), particularly `app/server/runtime.cpp`, `src/models/qwen3_asr/session.cpp`, `src/models/qwen3_forced_aligner/processor.cpp`, and its MUSA optimization report.

The contributor approved the native Qwen3-based classifier and persistent companion-model architecture before implementation. This design and its implementation target the private fork; they do not imply upstream submission or approval.
