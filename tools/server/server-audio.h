#pragma once

#include "common.h"
#include "json.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct server_audio_info {
    uint32_t sample_rate = 0;
    uint64_t frames = 0;
    double duration_ms() const { return 1000.0 * frames / sample_rate; }
};

struct server_audio_transcript {
    std::string text;
    std::string language;
};

struct server_audio_unavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

server_audio_info server_audio_read_wav_info(const std::vector<uint8_t> & audio);
std::string server_audio_language(const std::string & language);
server_audio_transcript server_audio_parse_transcript(const std::string & text, const std::string & language);
std::vector<std::string> server_audio_words(const std::string & text, const std::string & language);
common_json server_audio_timestamps(const std::vector<std::string> & words, const std::vector<int32_t> & classes, int64_t step_ms, uint32_t sample_rate);

class server_audio_aligner {
public:
    explicit server_audio_aligner(common_params & params);
    ~server_audio_aligner();
    void request_stop();
    common_json align(const std::vector<uint8_t> & audio, const server_audio_transcript & transcript, const std::function<bool()> & should_stop);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
