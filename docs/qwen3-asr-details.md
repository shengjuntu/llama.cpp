# Qwen3-ASR with forced alignment

`llama-server` can run Qwen3-ASR and Qwen3-ForcedAligner together through `POST /v1/audio/transcriptions/details`. The route recognizes speech, aligns the final recognized text, and returns word or character offsets on the original WAV timeline. Both stages use native GGML. The running server needs neither Python nor ONNX Runtime.

This implementation targets `shengjuntu/llama.cpp`, based on commit `5d806aa2575e01e126651fd69ab1ab6cefff861d`. Its multipart WAV response follows the corresponding audio.cpp details fields. See the [implementation design](development/qwen3-asr-details-plan.md).

## Convert the models

Download the complete original checkpoints [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), including tokenizer and configuration files. Install the repository's conversion requirements on the conversion machine.

From the repository root, with the checkpoints in `models/`:

```bash
python convert_hf_to_gguf.py models/Qwen3-ASR-0.6B --outtype f16 --outfile models/asr-f16.gguf
python convert_hf_to_gguf.py models/Qwen3-ASR-0.6B --mmproj --outtype f16 --outfile models/asr-mmproj-f16.gguf
python convert_hf_to_gguf.py models/Qwen3-ForcedAligner-0.6B --outtype f16 --outfile models/aligner-f16.gguf
python convert_hf_to_gguf.py models/Qwen3-ForcedAligner-0.6B --mmproj --outtype f16 --outfile models/aligner-mmproj-f16.gguf
```

Use this version of the converter for all four files. It adds the ASR identity flag and the aligner's class count, timestamp token, and time step. Existing audio.cpp GGUF files use a different layout and cannot be substituted for these files.

The aligner decoder has architecture `qwen3aligner` and a timestamp classification head. It is not a text generation model. F16 conversion keeps the classification head in F32. The native HF `Qwen3ASRForTokenClassification` namespace is also mapped; the validation scope for that layout is listed below.

Optional decoder quantization:

```bash
build/bin/llama-quantize models/aligner-f16.gguf models/aligner-q8_0.gguf Q8_0
```

Quantization can change timestamp classes. Keep the F16 files as the reference when evaluating another format or backend.

## Build and start

CPU build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build --target llama-server llama-quantize test-llama-archs test-chat -j4
```

Start one ASR model and its companion aligner:

```bash
build/bin/llama-server \
  --model models/asr-f16.gguf \
  --mmproj models/asr-mmproj-f16.gguf \
  --aligner-model models/aligner-f16.gguf \
  --aligner-mmproj models/aligner-mmproj-f16.gguf \
  --alias qwen3-asr --host 127.0.0.1 --port 32101 \
  --ctx-size 8192 --aligner-ctx-size 8192 --parallel 1 \
  --threads 4 --threads-batch 4 --gpu-layers 0 --no-mmproj-offload
```

For MUSA SDK 5.1.0, follow the existing [MUSA build instructions](build.md#musa), enabling `-DGGML_MUSA=ON` in a separate build directory. Select `MUSA_ARCHITECTURES` for the actual GPU if needed. Launch the resulting server with `--gpu-layers 99 --aligner-gpu-layers 99` and remove `--no-mmproj-offload`. This backend uses the same model graphs; its runtime correctness, memory use, and speed still require testing on the target device.

The companion options are server options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--aligner-model FILE` | absent | Aligner decoder GGUF. |
| `--aligner-mmproj FILE` | absent | Aligner audio encoder/projector GGUF. |
| `--aligner-ctx-size N` | 8192 | Aligner context capacity, subject to normal llama.cpp alignment. |
| `--aligner-gpu-layers N` | -1 | Inherit ASR decoder placement; 0 selects CPU; positive values select GPU layers. |

Environment equivalents are `LLAMA_ARG_ALIGNER_MODEL`, `LLAMA_ARG_ALIGNER_MMPROJ`, `LLAMA_ARG_ALIGNER_CTX_SIZE`, and `LLAMA_ARG_ALIGNER_GPU_LAYERS`. Thread counts and flash attention settings follow the ASR configuration. `--no-mmproj-offload` applies to both audio encoders. `--aligner-gpu-layers 0` also keeps the aligner projector on CPU.

Both aligner paths must be provided together. Startup checks the ASR and aligner identities. Sleep mode must be disabled (`--sleep-idle-seconds -1`, the default). Configure the companion on an inference server; configuring it directly on a model router is rejected.

## Request and response

```bash
curl --fail-with-body http://127.0.0.1:32101/v1/audio/transcriptions/details \
  -F model=qwen3-asr \
  -F file=@recording.wav \
  -F language=zh
```

Omit `language` or set it to `auto` for language detection. `prompt` is optional recognition context; `text` is accepted as its legacy alias. Neither field supplies the transcript for alignment. The aligner always receives the final ASR result.

`max_tokens` defaults to 512 and must be a positive integer. `temperature` defaults to 0 and must be finite and between 0 and 2. Only `response_format=json` and non-streaming requests are supported. The details route always performs alignment when the recognized text has alignment units.

Response shape, with a shortened word list:

```json
{
  "text": "Concord returned to its place amidst the tents.",
  "language": "English",
  "words": [
    {"word": "Concord", "start_sample": 8960, "end_sample": 19200, "confidence": 0.0}
  ],
  "sample_rate": 16000,
  "timing": {"wall_ms": 1600.0, "audio_duration_ms": 3505.0, "rtf": 0.45649}
}
```

The timing numbers above illustrate the schema. `wall_ms` measures processing after header/field validation through both inference stages, including queue waits. It excludes the network upload and response transfer. `rtf` is `wall_ms / audio_duration_ms`; there is no per-stage timing field yet.

Offsets count frames per channel at the original sample rate, including for stereo files. Chinese and Cantonese use individual CJK characters with Latin runs split at punctuation; other supported languages use whitespace-delimited units with punctuation removed. The original punctuation remains in `text`. `confidence=0.0` is audio.cpp's unavailable-confidence sentinel, not a probability.

Timestamp classes use the model's time step (80 ms for the published checkpoint). Postprocessing uses the official longest non-decreasing subsequence repair, followed by audio.cpp-compatible repair of zero-duration spans. It does not clamp the last offset to the WAV duration. Empty results omit `words` and `sample_rate`; speaker and sentence fields are not synthesized.

## Limits and errors

- WAV input only: RIFF PCM 8/16/24/32-bit or float 32/64-bit, including multichannel audio. JSON audio input, MP3 upload, VAD, and user-level chunk aggregation are outside this version.
- Audio must be non-empty and at most 300 seconds. Both contexts must fit their complete inputs. The internal 8-second encoder windows are part of model inference and do not divide the ASR transcript into separate requests.
- Alignment languages: `zh`, `yue`, `en`, `de`, `es`, `fr`, `it`, `pt`, `ru`, or their English names. Japanese and Korean need dedicated segmentation work and currently return 400. Detected languages outside this list also return an error.
- Invalid fields, unsupported languages, streaming details, and context overflow return 400. The implementation does not truncate alignment text.
- Missing companion configuration or a full/stopping alignment queue returns 503. Inference failures and ASR generation that stops before EOS return 500, with a specific error message. No partial alignment is returned as success.
- One persistent worker owns the aligner model, context, and its own encoder. It serializes alignment and permits at most 16 waiting alignment jobs. ASR uses the existing inference queue. Each alignment clears its KV state before and after evaluation.
- Disconnect and shutdown cancellation is checked between alignment chunks/batches. An already running backend graph finishes before cancellation takes effect. Router forwarding and cancellation under sustained load have not been validated.

## Validation

CPU validation used the original 0.6B checkpoints, F16 GGUF weights, an F32 classification head, and the official PyTorch implementation with float32 weights and eager attention. Tests used four CPU threads. The following alignment inputs produced identical prompt token sequences and timestamp argmax classes:

| Fixture | Duration | Prompt tokens | Matching timestamp classes |
| --- | --- | --- | --- |
| LibriSpeech `6930-75918-0000` | 3.505 s | 76 | 16 / 16 |
| LibriSpeech `6930-75918-0001`, crossing encoder windows | 14.225 s | 324 | 86 / 86 |
| SenseVoiceSmall `example/zh.mp3`, converted to 16 kHz PCM WAV | 5.616 s | 114 | 26 / 26 |
| First 0.9 s of the first fixture, with supplied word `Concord` | 0.900 s | 18 | 2 / 2 |

The first fixture exposed a pre-existing Qwen3 audio frontend issue: 52 padded audio tokens were passed onward where the reference uses 46. The fix drops the final centered STFT frame and removes padded CNN tokens before encoder self-attention. After the fix, its classification-score RMSE against the float32 reference was approximately 0.0111, with all 16 argmax classes matching. Matching classes do not imply bitwise-equal activations or parity on all recordings.

Real HTTP checks passed for Chinese/English recognition plus alignment, automatic language detection, silence, repeated and concurrent requests, 24 kHz stereo input, malformed WAV, invalid numeric fields, unsupported language, generation limits, both context limits, and missing companion configuration. Q8_0 alignment passed the first English HTTP fixture with the same word offsets as F16.

An additional 32 kHz stereo resampling round trip kept the transcript but changed one timestamp by one 80 ms bin. Input resampling is therefore not claimed to preserve all class predictions. Offsets still use the original sample rate; the unit test also checks exact 44.1 kHz conversion from known classes.

The native HF namespace was checked by remapping the original weights into that layout and converting them with the published native configuration. All 311 decoder tensors and 398 projector tensors matched the original conversion. A separate native HF weight download was not used for an end-to-end test.

Regression commands:

```bash
build/bin/test-llama-archs -a qwen3aligner -s 42
build/bin/test-llama-archs -a qwen3 -s 42
build/bin/test-chat
```

The architecture test covers classifier width differing from both vocabulary and hidden size, plus model save/load including timestamp metadata. The chat test covers transcript protocol parsing, mixed Chinese/Latin units, sample-rate conversion, and timestamp repair. Ordinary perplexity and text-generation tests do not apply to the timestamp classifier.

MUSA SDK 5.1.0 runtime tests and comparison against the deployed audio.cpp service remain outstanding. Measure warmed combined RTF, peak memory, and timestamp differences on the target hardware before making a performance claim.
