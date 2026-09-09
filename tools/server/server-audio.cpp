#include "server-audio.h"

#include "llama-cpp.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "unicode.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {

std::string lower_ascii(std::string value) {
    for (char & c : value) {
        if (c >= 'A' && c <= 'Z') { c += 'a' - 'A'; }
    }
    return value;
}

std::string trim(const std::string & text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

bool is_cjk(uint32_t cp) {
    return (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x4e00 && cp <= 0x9fff) ||
           (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x20000 && cp <= 0x2a6df) ||
           (cp >= 0x2a700 && cp <= 0x2b73f) || (cp >= 0x2b740 && cp <= 0x2b81f) ||
           (cp >= 0x2b820 && cp <= 0x2ceaf);
}

bool keep_character(uint32_t cp) {
    return cp == '\'' || (cp >= '0' && cp <= '9') || (cp >= 'A' && cp <= 'Z') ||
           (cp >= 'a' && cp <= 'z') || is_cjk(cp) || (cp >= 0x00c0 && cp <= 0x02af) ||
           (cp >= 0x0370 && cp <= 0x052f) || (cp >= 0x0e00 && cp <= 0x0e7f) ||
           (cp >= 0x3040 && cp <= 0x30ff) || (cp >= 0xac00 && cp <= 0xd7af);
}

std::string model_metadata(const llama_model * model, const char * key) {
    char value[128];
    const int n = llama_model_meta_val_str(model, key, value, sizeof(value));
    if (n < 0 || n >= int(sizeof(value))) {
        throw std::runtime_error(std::string("Missing or invalid model metadata: ") + key);
    }
    return std::string(value, n);
}

std::vector<int64_t> monotonic_timestamps(const std::vector<int64_t> & values) {
    if (values.empty()) {
        return {};
    }
    const size_t n = values.size();
    std::vector<size_t> length(n, 1), parent(n, n);
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (values[j] <= values[i] && length[j] + 1 > length[i]) {
                length[i] = length[j] + 1;
                parent[i] = j;
            }
        }
        if (length[i] > length[best]) {
            best = i;
        }
    }
    std::vector<bool> anchor(n, false);
    for (size_t i = best; i != n; i = parent[i]) {
        anchor[i] = true;
    }
    auto out = values;
    for (size_t first = 0; first < n;) {
        if (anchor[first]) {
            ++first;
            continue;
        }
        size_t end = first;
        while (end < n && !anchor[end]) {
            ++end;
        }
        for (size_t i = first; i < end; ++i) {
            if (first == 0) {
                out[i] = out[end];
            } else if (end == n) {
                out[i] = out[first - 1];
            } else if (end - first <= 2) {
                out[i] = i - first + 1 <= end - i ? out[first - 1] : out[end];
            } else {
                const double step = double(out[end] - out[first - 1]) / (end - first + 1);
                out[i] = int64_t(out[first - 1] + step * (i - first + 1));
            }
        }
        first = end;
    }
    return out;
}

} // namespace

server_audio_info server_audio_read_wav_info(const std::vector<uint8_t> & audio) {
    auto u16 = [&](size_t i) { return uint32_t(audio[i]) | (uint32_t(audio[i + 1]) << 8); };
    auto u32 = [&](size_t i) { return u16(i) | (u16(i + 2) << 16); };
    if (audio.size() < 12 || std::memcmp(audio.data(), "RIFF", 4) || std::memcmp(audio.data() + 8, "WAVE", 4)) {
        throw std::invalid_argument("details requires a RIFF WAV file with PCM or float samples");
    }
    const uint64_t limit = uint64_t(u32(4)) + 8;
    if (limit > audio.size() || limit < 12) {
        throw std::invalid_argument("Truncated WAV file");
    }
    uint32_t rate = 0, block = 0;
    uint64_t bytes = 0;
    bool have_data = false;
    for (size_t i = 12; i + 8 <= limit;) {
        const uint32_t size = u32(i + 4);
        const size_t data = i + 8;
        if (size > limit - data) {
            throw std::invalid_argument("Truncated WAV chunk");
        }
        if (!std::memcmp(audio.data() + i, "fmt ", 4)) {
            if (size < 16) {
                throw std::invalid_argument("Invalid WAV format chunk");
            }
            uint32_t format = u16(data);
            if (format == 0xfffe && size >= 40) {
                format = u16(data + 24);
            }
            const uint32_t channels = u16(data + 2), bits = u16(data + 14);
            rate = u32(data + 4);
            block = u16(data + 12);
            const bool valid_samples = (format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (format == 3 && (bits == 32 || bits == 64));
            if (!valid_samples || !channels || channels > 32 || !rate || rate > 384000 || block != channels * (bits / 8) || u32(data + 8) != rate * block) {
                throw std::invalid_argument("Unsupported or invalid WAV format");
            }
        } else if (!std::memcmp(audio.data() + i, "data", 4)) {
            if (have_data) {
                throw std::invalid_argument("Multiple WAV data chunks are not supported");
            }
            bytes = size;
            have_data = true;
        }
        i = data + size + (size & 1);
    }
    if (!rate || !block || !have_data || !bytes || bytes % block) {
        throw std::invalid_argument("WAV must contain complete, non-empty audio frames");
    }
    server_audio_info info{rate, bytes / block};
    if (info.frames > uint64_t(rate) * 300) {
        throw std::invalid_argument("details supports at most 300 seconds without chunking");
    }
    return info;
}

std::string server_audio_language(const std::string & language) {
    const std::string value = lower_ascii(trim(language));
    if (value.empty() || value == "auto" || value == "none") {
        return "";
    }
    static const std::pair<const char *, const char *> languages[] = {
        {"zh", "Chinese"}, {"yue", "Cantonese"}, {"en", "English"}, {"de", "German"},
        {"es", "Spanish"}, {"fr", "French"}, {"it", "Italian"}, {"pt", "Portuguese"}, {"ru", "Russian"},
    };
    for (const auto & item : languages) {
        if (value == item.first || value == lower_ascii(item.second)) {
            return item.second;
        }
    }
    throw std::invalid_argument("Unsupported alignment language: " + language + "; supported: zh, yue, en, de, es, fr, it, pt, ru");
}

server_audio_transcript server_audio_parse_transcript(const std::string & text, const std::string & language) {
    server_audio_transcript out{trim(text), server_audio_language(language)};
    const auto separator = out.text.find("<asr_text>");
    if (separator != std::string::npos) {
        const std::string prefix = trim(out.text.substr(0, separator));
        if (prefix.compare(0, 9, "language ") == 0) {
            out.language = server_audio_language(prefix.substr(9));
        }
        out.text = trim(out.text.substr(separator + 10));
    }
    if (out.text == "language None") {
        out = {};
    }
    if (!out.text.empty() && out.language.empty()) {
        throw std::runtime_error("ASR did not return a language; supply the language field for alignment");
    }
    return out;
}

std::vector<std::string> server_audio_words(const std::string & text, const std::string & language) {
    const auto normalized = server_audio_language(language);
    const bool chinese = normalized == "Chinese" || normalized == "Cantonese";
    std::vector<std::string> words;
    std::string current;
    auto flush = [&]() {
        if (!current.empty()) {
            words.push_back(std::move(current));
            current.clear();
        }
    };
    for (size_t i = 0; i < text.size();) {
        const auto cp = common_parse_utf8_codepoint(text, i);
        if (cp.status != utf8_parse_result::SUCCESS) {
            throw std::invalid_argument("Invalid UTF-8 in the transcript");
        }
        const auto piece = text.substr(i, cp.bytes_consumed);
        if (is_cjk(cp.codepoint)) {
            flush();
            words.push_back(piece);
        } else if (keep_character(cp.codepoint)) {
            current += piece;
        } else if (chinese || cp.codepoint == ' ' || cp.codepoint == '\t' || cp.codepoint == '\r' || cp.codepoint == '\n') {
            flush();
        }
        i += cp.bytes_consumed;
    }
    flush();
    return words;
}

common_json server_audio_timestamps(const std::vector<std::string> & words, const std::vector<int32_t> & classes, int64_t step_ms, uint32_t sample_rate) {
    if (classes.size() != 2 * words.size() || step_ms <= 0 || step_ms > 1000 || sample_rate == 0 || sample_rate > 384000) {
        throw std::runtime_error("Invalid timestamp prediction dimensions or time scale");
    }
    std::vector<int64_t> times;
    for (int32_t id : classes) {
        if (id < 0) {
            throw std::runtime_error("Negative timestamp class");
        }
        times.push_back(int64_t(id) * step_ms);
    }
    times = monotonic_timestamps(times);
    for (auto & value : times) {
        value = (value * sample_rate + 500) / 1000;
    }
    common_json out = common_json::array();
    const int64_t minimum_span = std::max<int64_t>(1, (step_ms * sample_rate + 500) / 1000);
    int64_t previous_end = 0;
    for (size_t i = 0; i < words.size(); ++i) {
        const int64_t start = std::max(previous_end, times[2 * i]);
        int64_t end = times[2 * i + 1];
        if (end <= start) {
            end = start + minimum_span;
            if (i + 1 < words.size() && times[2 * i + 2] > start) {
                end = std::min(end, times[2 * i + 2]);
            }
        }
        out.push_back({{"word", words[i]}, {"start_sample", start}, {"end_sample", end}, {"confidence", 0.0}});
        previous_end = end;
    }
    return out;
}

struct server_audio_aligner::impl {
    struct job {
        std::vector<uint8_t> audio;
        server_audio_transcript transcript;
        std::atomic<bool> cancelled{false};
        std::promise<common_json> result;
    };

    llama_model_ptr model;
    llama_context_ptr context;
    mtmd::context_ptr projector;
    llama_token timestamp_token;
    int64_t step_ms;
    int32_t n_classes;
    std::atomic<bool> stopping{false};
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::shared_ptr<job>> queue;
    std::thread worker;

    explicit impl(common_params & params) {
        auto mp = common_model_params_to_llama(params);
        if (params.aligner_n_gpu_layers != -1) {
            mp.n_gpu_layers = params.aligner_n_gpu_layers;
        }
        model.reset(llama_model_load_from_file(params.aligner_model.c_str(), mp));
        if (!model || model_metadata(model.get(), "general.architecture") != "qwen3aligner") {
            throw std::runtime_error("--aligner-model requires a qwen3aligner GGUF checkpoint");
        }
        timestamp_token = std::stoi(model_metadata(model.get(), "qwen3aligner.timestamp_token_id"));
        const double scale = std::stod(model_metadata(model.get(), "qwen3aligner.timestamp_segment_time"));
        if (!std::isfinite(scale) || scale <= 0 || scale > 1000 || std::floor(scale) != scale) {
            throw std::runtime_error("Invalid aligner timestamp time step");
        }
        step_ms = int64_t(scale);
        n_classes = llama_model_n_embd_out(model.get());
        const auto marker = common_tokenize(llama_model_get_vocab(model.get()), "<timestamp>", false, true);
        if (marker.size() != 1 || marker[0] != timestamp_token) {
            throw std::runtime_error("Aligner timestamp token does not match its tokenizer");
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = params.aligner_n_ctx;
        cp.n_batch = std::min({512, params.n_batch, params.n_ubatch});
        cp.n_ubatch = cp.n_batch;
        cp.n_threads = params.cpuparams.n_threads;
        cp.n_threads_batch = params.cpuparams_batch.n_threads;
        cp.flash_attn_type = params.flash_attn_type;
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        context.reset(llama_init_from_model(model.get(), cp));
        if (!context) {
            throw std::runtime_error("Failed to initialize the aligner context");
        }
        auto pp = mtmd_context_params_default();
        pp.use_gpu = params.mmproj_use_gpu && params.aligner_n_gpu_layers != 0;
        pp.device = params.mmproj_device;
        pp.n_threads = params.cpuparams.n_threads;
        pp.flash_attn_type = params.flash_attn_type;
        pp.warmup = false;
        projector.reset(mtmd_init_from_file(params.aligner_mmproj.c_str(), model.get(), pp));
        if (!projector || !mtmd_support_audio(projector.get()) || mtmd_decode_use_mrope(projector.get())) {
            throw std::runtime_error("Failed to initialize the aligner audio projector");
        }
        worker = std::thread([this]() { run(); });
    }

    ~impl() {
        stop();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void stop() {
        stopping = true;
        ready.notify_all();
    }

    void check_cancelled(const job & task) const {
        if (stopping || task.cancelled) {
            throw std::runtime_error("Alignment cancelled");
        }
    }

    common_json evaluate(job & task) {
        check_cancelled(task);
        const auto info = server_audio_read_wav_info(task.audio);
        const auto words = server_audio_words(task.transcript.text, task.transcript.language);
        if (words.empty()) {
            return common_json::array();
        }
        auto wrapper = mtmd_helper_bitmap_init_from_buf(projector.get(), task.audio.data(), task.audio.size(), false, mtmd_helper_init_opt_default());
        mtmd::bitmap_ptr bitmap(wrapper.bitmap);
        if (!bitmap) {
            throw std::invalid_argument("Failed to decode alignment audio");
        }
        mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
        const mtmd_bitmap * bitmap_raw = bitmap.get();
        mtmd_input_part part{nullptr, bitmap_raw};
        const mtmd_input_part * parts[] = {&part};
        if (mtmd_tokenize_from_parts(projector.get(), chunks.get(), parts, 1, false)) {
            throw std::runtime_error("Failed to preprocess alignment audio");
        }
        std::vector<llama_token> text_tokens;
        for (const auto & word : words) {
            auto tokens = common_tokenize(llama_model_get_vocab(model.get()), word, false, false);
            text_tokens.insert(text_tokens.end(), tokens.begin(), tokens.end());
            text_tokens.push_back(timestamp_token);
            text_tokens.push_back(timestamp_token);
        }
        if (mtmd_helper_get_n_pos(chunks.get()) + text_tokens.size() > llama_n_ctx(context.get())) {
            throw std::invalid_argument("Audio and transcript exceed --aligner-ctx-size; no text was truncated");
        }
        llama_memory_clear(llama_get_memory(context.get()), true);
        struct clear_context {
            llama_context * ctx;
            ~clear_context() { llama_memory_clear(llama_get_memory(ctx), true); }
        } cleanup{context.get()};

        llama_pos position = 0;
        const int32_t batch_size = llama_n_batch(context.get());
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            check_cancelled(task);
            if (mtmd_helper_eval_chunk_single(projector.get(), context.get(), mtmd_input_chunks_get(chunks.get(), i), position, 0, batch_size, false, &position)) {
                throw std::runtime_error("Failed to evaluate alignment audio");
            }
        }
        std::vector<int32_t> predictions;
        auto batch = llama_batch_init(batch_size, 0, 1);
        struct free_batch {
            llama_batch & value;
            ~free_batch() { llama_batch_free(value); }
        } batch_cleanup{batch};
        for (size_t off = 0; off < text_tokens.size(); off += batch_size) {
            check_cancelled(task);
            common_batch_clear(batch);
            const size_t count = std::min<size_t>(batch_size, text_tokens.size() - off);
            for (size_t i = 0; i < count; ++i) {
                common_batch_add(batch, text_tokens[off + i], position++, {0}, true);
            }
            if (llama_decode(context.get(), batch)) {
                throw std::runtime_error("Failed to evaluate alignment text");
            }
            for (size_t i = 0; i < count; ++i) {
                if (text_tokens[off + i] != timestamp_token) {
                    continue;
                }
                const float * scores = llama_get_embeddings_ith(context.get(), i);
                if (!scores || !std::all_of(scores, scores + n_classes, [](float v) { return std::isfinite(v); })) {
                    throw std::runtime_error("Aligner produced missing or non-finite timestamp scores");
                }
                predictions.push_back(std::max_element(scores, scores + n_classes) - scores);
            }
        }
        check_cancelled(task);
        return server_audio_timestamps(words, predictions, step_ms, info.sample_rate);
    }

    void run() {
        for (;;) {
            std::shared_ptr<job> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock, [&]() { return stopping || !queue.empty(); });
                if (queue.empty()) {
                    return;
                }
                task = queue.front();
                queue.pop_front();
            }
            try {
                task->result.set_value(evaluate(*task));
            } catch (...) {
                task->result.set_exception(std::current_exception());
            }
        }
    }
};

server_audio_aligner::server_audio_aligner(common_params & params) : pimpl(new impl(params)) {}
server_audio_aligner::~server_audio_aligner() = default;

void server_audio_aligner::request_stop() {
    pimpl->stop();
}

common_json server_audio_aligner::align(const std::vector<uint8_t> & audio, const server_audio_transcript & transcript, const std::function<bool()> & should_stop) {
    auto task = std::make_shared<impl::job>();
    task->audio = audio;
    task->transcript = transcript;
    auto result = task->result.get_future();
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        if (pimpl->stopping || pimpl->queue.size() >= 16) {
            throw server_audio_unavailable("Alignment worker is stopping or its queue is full");
        }
        pimpl->queue.push_back(task);
    }
    pimpl->ready.notify_one();
    while (result.wait_for(std::chrono::milliseconds(25)) != std::future_status::ready) {
        if (should_stop()) {
            task->cancelled = true;
            return common_json::array();
        }
    }
    return result.get();
}
