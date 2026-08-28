#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImplContext.h"
#include <algorithm>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

constexpr size_t kMaxSampleKeyCount  = 4;
constexpr size_t kMaxSampleKeyLength = 128;

const char* failureSourceToString(CacheLoadFailureSource source) {
    switch (source) {
        case CacheLoadFailureSource::TIMER_EXPIRED:
            return "timer_expired";
        case CacheLoadFailureSource::REQUEST_BLOCK_BUFFER_CLOSED:
            return "request_block_buffer_closed";
        case CacheLoadFailureSource::WATCH_REGISTRATION_FAILED:
            return "watch_registration_failed";
        case CacheLoadFailureSource::INVALID_BLOCK:
            return "invalid_block";
        case CacheLoadFailureSource::RESPONSE_WRITE_FAILED:
            return "response_write_failed";
        case CacheLoadFailureSource::UNKNOWN:
        default:
            return "unknown";
    }
}

int64_t ageMs(int64_t timestamp_us, int64_t now_us) {
    return timestamp_us > 0 ? (now_us - timestamp_us) / 1000 : -1;
}

uint64_t hashKeys(const std::vector<std::string>& keys) {
    constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
    constexpr uint64_t kFnvPrime       = 1099511628211ULL;
    uint64_t           hash            = kFnvOffsetBasis;
    for (const auto& key : keys) {
        const auto size = static_cast<uint64_t>(key.size());
        for (size_t i = 0; i < sizeof(size); ++i) {
            hash ^= static_cast<unsigned char>((size >> (i * 8)) & 0xff);
            hash *= kFnvPrime;
        }
        for (const unsigned char ch : key) {
            hash ^= ch;
            hash *= kFnvPrime;
        }
    }
    return hash;
}

void appendKey(std::ostringstream& stream, const std::string& key) {
    if (key.size() <= kMaxSampleKeyLength) {
        stream << key;
    } else {
        stream << key.substr(0, kMaxSampleKeyLength - 3) << "...";
    }
}

void appendKeySet(std::ostringstream& stream, const char* name, std::vector<std::string> keys) {
    std::sort(keys.begin(), keys.end());
    stream << ", " << name << "_count=" << keys.size() << ", " << name << "_hash=" << std::hex
           << std::setw(16) << std::setfill('0') << hashKeys(keys) << std::dec << std::setfill(' ') << ", sample_"
           << name << "_keys=[";
    const auto sampled_count = std::min(keys.size(), kMaxSampleKeyCount);
    for (size_t i = 0; i < sampled_count; ++i) {
        if (i > 0) {
            stream << ",";
        }
        appendKey(stream, keys[i]);
    }
    stream << "], omitted_" << name << "_keys=" << keys.size() - sampled_count;
}

std::string formatBlockBufferDebugInfo(const std::string&                                  request_id,
                                       const std::optional<RequestBlockBufferDebugInfo>& debug_info) {
    if (!debug_info.has_value()) {
        return "request id: " + request_id + " not found or expired";
    }
    const auto& info = debug_info.value();
    std::ostringstream stream;
    stream << "request id: " << request_id << ", blocks count: " << info.block_keys.size()
           << ", block adds: " << info.block_add_count << ", watch funcs: " << info.watch_func_count
           << ", watch triggers: " << info.watch_trigger_count
           << ", watch callback dispatches: " << info.watch_callback_dispatch_count
           << ", watch callback completes: " << info.watch_callback_complete_count
           << ", watch callback inflight: " << info.watch_callback_inflight_count;
    return stream.str();
}

}  // namespace

CacheStoreServiceImplContext::CacheStoreServiceImplContext(
    const CacheLoadRequest*                                      request,
    CacheLoadResponse*                                           response,
    const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
    ::google::protobuf::Closure*                                 done,
    const std::shared_ptr<RequestBlockBufferStore>&              request_block_buffer_store):
    request_(request),
    context_create_time_us_(currentTimeUs()),
    request_send_start_time_us_(request->request_send_start_time_us()),
    total_block_count_(request_->blocks_size()),
    request_id_(request_->requestid()),
    peer_ip_(request->client_ip()),
    partition_count_(request->partition_count() == 0 ? 1 : request->partition_count()),  // compatible with old version
    partition_id_(request->partition_id()),
    response_(response),
    collector_(collector),
    done_(done),
    request_block_buffer_store_(request_block_buffer_store),
    write_cnt_(0) {
    // init set unloaded blocks
    std::unique_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
    for (int i = 0; i < request_->blocks_size(); i++) {
        const auto& key = request_->blocks(i).key();
        if (unloaded_blocks_.find(key) != unloaded_blocks_.end() && duplicate_block_key_samples_.size() < 16) {
            duplicate_block_key_samples_.push_back(key);
        }
        unloaded_blocks_[key] = std::make_shared<BlockBufferInfo>(request_->blocks(i));
    }
    unique_block_count_ = static_cast<uint32_t>(unloaded_blocks_.size());
}

std::shared_ptr<BlockBufferInfo> CacheStoreServiceImplContext::getAndEraseUnLoadedBlock(const std::string& block_key) {
    RTP_LLM_PROFILE_FUNCTION();
    std::unique_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
    auto                                it = unloaded_blocks_.find(block_key);
    if (it == unloaded_blocks_.end()) {
        return nullptr;
    }
    if (unloaded_blocks_.size() == total_block_count_) {
        collector_->markFirstBlockReady();
    }

    auto block_info = it->second;
    unloaded_blocks_.erase(it);
    last_block_match_time_us_ = currentTimeUs();

    if (unloaded_blocks_.empty()) {
        collector_->markAllBlocksReady();
    }
    return block_info;
}

std::string CacheStoreServiceImplContext::getLoadProgressDebugInfo() {
    auto request_block_buffer_store = request_block_buffer_store_.lock();
    auto debug_info                 = request_block_buffer_store
                                          ? request_block_buffer_store->getDebugInfoOnRequest(request_id_)
                                          : std::optional<RequestBlockBufferDebugInfo>();
    return getLoadProgressDebugInfo(debug_info);
}

std::string CacheStoreServiceImplContext::getLoadProgressDebugInfo(
    const std::optional<RequestBlockBufferDebugInfo>& block_buffer_debug_info) {
    const auto now_us = currentTimeUs();

    std::vector<std::string> remaining_unloaded_keys;
    {
        std::shared_lock<std::shared_mutex> lock(unloaded_blocks_mutex_);
        remaining_unloaded_keys.reserve(unloaded_blocks_.size());
        for (const auto& entry : unloaded_blocks_) {
            remaining_unloaded_keys.push_back(entry.first);
        }
    }
    const auto remaining_unloaded_count = remaining_unloaded_keys.size();
    const auto ready_count =
        unique_block_count_ >= remaining_unloaded_count ? unique_block_count_ - remaining_unloaded_count : 0;
    const auto write_done_count = static_cast<uint32_t>(std::max(write_cnt_.load(), 0));
    const auto pending_write_count = ready_count >= write_done_count ? ready_count - write_done_count : 0;

    std::unordered_set<std::string> expected_keys;
    expected_keys.reserve(unique_block_count_);
    for (const auto& block : request_->blocks()) {
        expected_keys.insert(block.key());
    }
    std::unordered_set<std::string> stored_keys;
    if (block_buffer_debug_info.has_value()) {
        stored_keys.insert(block_buffer_debug_info->block_keys.begin(), block_buffer_debug_info->block_keys.end());
    }
    std::vector<std::string> missing_in_store_keys;
    std::vector<std::string> present_but_unconsumed_keys;
    std::vector<std::string> unexpected_in_store_keys;
    if (block_buffer_debug_info.has_value()) {
        for (const auto& key : expected_keys) {
            if (stored_keys.find(key) == stored_keys.end()) {
                missing_in_store_keys.push_back(key);
            }
        }
        for (const auto& key : remaining_unloaded_keys) {
            if (stored_keys.find(key) != stored_keys.end()) {
                present_but_unconsumed_keys.push_back(key);
            }
        }
        for (const auto& key : stored_keys) {
            if (expected_keys.find(key) == expected_keys.end()) {
                unexpected_in_store_keys.push_back(key);
            }
        }
    }

    std::ostringstream stream;
    stream << "{failure_source=" << failureSourceToString(failure_source_.load()) << ", total=" << total_block_count_
           << ", unique_expected=" << unique_block_count_
           << ", duplicate_request_keys=" << total_block_count_ - unique_block_count_ << ", ready=" << ready_count
           << ", remaining_unloaded=" << remaining_unloaded_count << ", write_done=" << write_done_count
           << ", pending_write=" << pending_write_count << ", server_context_age_ms="
           << ageMs(context_create_time_us_, now_us) << ", client_request_age_ms="
           << ageMs(request_send_start_time_us_, now_us) << ", watch_register_result=" << watch_register_result_.load()
           << ", watch_register_age_ms=" << ageMs(watch_register_time_us_.load(), now_us)
           << ", context_watch_callback_enter=" << watch_callback_enter_count_.load()
           << ", context_watch_callback_exit=" << watch_callback_exit_count_.load()
           << ", context_watch_callback_inflight=" << watch_callback_inflight_count_.load()
           << ", context_watch_callback_blocks=" << watch_callback_block_count_.load()
           << ", last_context_watch_callback_age_ms=" << ageMs(last_watch_callback_time_us_.load(), now_us)
           << ", last_block_match_age_ms=" << ageMs(last_block_match_time_us_.load(), now_us)
           << ", last_response_write_age_ms=" << ageMs(last_response_write_time_us_.load(), now_us)
           << ", block_buffer_found=" << block_buffer_debug_info.has_value();
    if (block_buffer_debug_info.has_value()) {
        const auto& info = block_buffer_debug_info.value();
        stream << ", stored_unique=" << info.block_keys.size() << ", store_block_adds=" << info.block_add_count
               << ", store_watch_funcs=" << info.watch_func_count
               << ", store_watch_triggers=" << info.watch_trigger_count
               << ", store_watch_callback_dispatches=" << info.watch_callback_dispatch_count
               << ", store_watch_callback_completes=" << info.watch_callback_complete_count
               << ", store_watch_callback_inflight=" << info.watch_callback_inflight_count
               << ", last_store_block_add_age_ms=" << ageMs(info.last_block_add_time_us, now_us)
               << ", last_store_watch_register_age_ms=" << ageMs(info.last_watch_register_time_us, now_us)
               << ", last_store_watch_trigger_age_ms=" << ageMs(info.last_watch_trigger_time_us, now_us)
               << ", last_store_watch_callback_complete_age_ms="
               << ageMs(info.last_watch_callback_complete_time_us, now_us);
        appendKeySet(stream, "missing_in_store", std::move(missing_in_store_keys));
        appendKeySet(stream, "present_but_unconsumed", std::move(present_but_unconsumed_keys));
        appendKeySet(stream, "unexpected_in_store", std::move(unexpected_in_store_keys));
    }
    appendKeySet(stream, "unloaded", std::move(remaining_unloaded_keys));
    appendKeySet(stream, "duplicate", duplicate_block_key_samples_);
    stream << "}";
    return stream.str();
}

void CacheStoreServiceImplContext::markWatchRegisterAttempt() {
    watch_register_time_us_ = currentTimeUs();
    watch_register_result_  = -1;
}

void CacheStoreServiceImplContext::markWatchRegisterResult(bool success) {
    watch_register_result_ = success ? 1 : 0;
}

void CacheStoreServiceImplContext::markWatchCallbackBegin(size_t block_count) {
    ++watch_callback_enter_count_;
    ++watch_callback_inflight_count_;
    watch_callback_block_count_.fetch_add(block_count);
    last_watch_callback_time_us_ = currentTimeUs();
}

void CacheStoreServiceImplContext::markWatchCallbackEnd() {
    --watch_callback_inflight_count_;
    ++watch_callback_exit_count_;
    last_watch_callback_time_us_ = currentTimeUs();
}

void CacheStoreServiceImplContext::runSuccess(bool direct_write) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("request [%s] run success", request_id_.c_str());
    bool expected = false;
    if (!done_run_.compare_exchange_strong(expected, true)) {
        return;
    }

    stopTimer();

    // run success, set response
    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        if (response_ != nullptr) {
            response_->set_error_code(KvCacheStoreServiceErrorCode::EC_SUCCESS);
            response_->set_response_send_start_time_us(currentTimeUs());
            response_->set_direct_write_response(direct_write);
            response_ = nullptr;
        }
    }

    collector_->markEnd(true);
    // call callback
    if (done_) {
        done_->Run();
        done_ = nullptr;
    }
}

void CacheStoreServiceImplContext::runFailed(KvCacheStoreServiceErrorCode error_code,
                                             CacheLoadFailureSource       failure_source) {
    RTP_LLM_PROFILE_FUNCTION();
    bool expected = false;
    if (!done_run_.compare_exchange_strong(expected, true)) {
        return;
    }
    failure_source_ = failure_source;

    stopTimer();

    auto request_block_buffer_store = request_block_buffer_store_.lock();
    auto block_buffer_debug_info     = request_block_buffer_store
                                           ? request_block_buffer_store->getDebugInfoOnRequest(request_id_)
                                           : std::optional<RequestBlockBufferDebugInfo>();
    const auto load_progress_debug_info = getLoadProgressDebugInfo(block_buffer_debug_info);
    if (request_block_buffer_store) {
        RTP_LLM_LOG_WARNING(
            "cache store service load failed, request %s from [%s], error code is %d, block buffer is %s, load "
            "progress is %s",
            request_id_.c_str(),
            peer_ip_.c_str(),
            error_code,
            formatBlockBufferDebugInfo(request_id_, block_buffer_debug_info).c_str(),
            load_progress_debug_info.c_str());
    } else {
        RTP_LLM_LOG_WARNING(
            "cache store service load failed, request %s from [%s], error code is %d, block buffer is null, load "
            "progress is %s",
            request_id_.c_str(),
            peer_ip_.c_str(),
            error_code,
            load_progress_debug_info.c_str());
    }

    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        if (response_ != nullptr) {
            response_->clear_blocks();
            response_->set_error_code(error_code);
            response_->set_response_send_start_time_us(currentTimeUs());
            response_ = nullptr;
        }
    }

    collector_->markEnd(false);
    if (done_) {
        RTP_LLM_LOG_WARNING("cache store service failure completion begin, request %s, failure source %s",
                            request_id_.c_str(),
                            failureSourceToString(failure_source_.load()));
        done_->Run();
        RTP_LLM_LOG_WARNING("cache store service failure completion end, request %s, failure source %s",
                            request_id_.c_str(),
                            failureSourceToString(failure_source_.load()));
        done_ = nullptr;
    } else {
        RTP_LLM_LOG_WARNING("cache store service failure completion callback is null, request %s, failure source %s",
                            request_id_.c_str(),
                            failureSourceToString(failure_source_.load()));
    }
}

void CacheStoreServiceImplContext::stopTimer() {
    if (auto timer_shared_ptr = timer_.lock()) {
        timer_shared_ptr->stop();
        timer_shared_ptr.reset();
    }
}

}  // namespace rtp_llm
