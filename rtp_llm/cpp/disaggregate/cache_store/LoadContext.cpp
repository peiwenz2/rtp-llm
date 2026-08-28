#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <sstream>

namespace rtp_llm {

SyncContext::SyncContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load):
    cache_store_(cache_store), combine_load_(combine_load) {}

void SyncContext::call(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                       int64_t                                                 timeout_ms,
                       CheckCancelFunc                                         check_cancel_func) {
    if (request_block_buffers.empty()) {
        return;
    }

    auto cache_store = cache_store_.lock();
    if (cache_store == nullptr) {
        error_info_ = ErrorInfo(ErrorCode::UNKNOWN_ERROR, ErrorCodeToString(ErrorCode::UNKNOWN_ERROR));
        RTP_LLM_LOG_WARNING("load failed, cache store is nullptr");
        return;
    }

    start_time_ms_     = autil::TimeUtility::currentTimeInMilliSeconds();
    deadline_ms_       = start_time_ms_ + timeout_ms;
    check_cancel_func_ = check_cancel_func;

    if (combine_load_) {  // for rdma only call rpc once
        auto new_buffer = std::make_shared<RequestBlockBuffer>(request_block_buffers[0]->getRequestId());
        for (auto& request_block_buffer : request_block_buffers) {
            auto blocks = request_block_buffer->getBlocks();
            for (auto& [_, block] : blocks) {
                new_buffer->addBlock(block);
            }
        }
        request_block_buffers_ = {new_buffer};
    } else {
        request_block_buffers_ = request_block_buffers;
    }

    expect_layer_cnt_ = request_block_buffers_.size();
    completed_request_buffers_.clear();
    duplicate_completion_count_ = 0;

    for (auto& request_block_buffer : request_block_buffers_) {
        if (!doCall(request_block_buffer, timeout_ms)) {
            updateResult(false, CacheStoreErrorCode::InvalidParams, request_block_buffer);
        }
    }
}

void SyncContext::updateResult(bool                                       success,
                               CacheStoreErrorCode                        ec,
                               const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto completion_inserted = completed_request_buffers_.insert(request_block_buffer.get()).second;
    if (!completion_inserted) {
        ++duplicate_completion_count_;
    }
    const auto done_layer_count = ++done_layer_cnt_;
    if (!success) {
        auto error_code = transCacheStoreErrorCode(ec);
        error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
        RTP_LLM_LOG_WARNING("request %s call finished, state:[%s], error code[%s], cost time %ldms, completion "
                            "progress %d/%d, duplicate completion %d",
                            request_block_buffer->getRequestKey().c_str(),
                            success ? "success" : "failed",
                            CacheStoreErrorCodeToString(ec).c_str(),
                            autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_,
                            done_layer_count,
                            expect_layer_cnt_,
                            !completion_inserted);
    } else {
        RTP_LLM_LOG_DEBUG("request %s call finished, state:[%s], cost time %ldms",
                          request_block_buffer->getRequestKey().c_str(),
                          success ? "success" : "failed",
                          autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_);
    }

    if (done_layer_count == expect_layer_cnt_) {
        cond_.notify_all();
    }
}

std::string SyncContext::getPendingRequestsDebugInfoLocked() const {
    constexpr size_t kMaxPendingRequestSamples = 16;
    constexpr size_t kMaxRequestKeyLength       = 128;
    const auto       now_us                     = currentTimeUs();
    std::ostringstream stream;
    size_t             pending_count = 0;
    size_t             sampled_count = 0;
    stream << "{done_layers=" << done_layer_cnt_.load() << ", expected_layers=" << expect_layer_cnt_
           << ", duplicate_completions=" << duplicate_completion_count_ << ", sample_pending_request_keys=[";
    for (const auto& request_block_buffer : request_block_buffers_) {
        if (completed_request_buffers_.find(request_block_buffer.get()) != completed_request_buffers_.end()) {
            continue;
        }
        ++pending_count;
        if (sampled_count >= kMaxPendingRequestSamples) {
            continue;
        }
        if (sampled_count++ > 0) {
            stream << ",";
        }
        const auto& request_key = request_block_buffer->getRequestKey();
        if (request_key.size() <= kMaxRequestKeyLength) {
            stream << request_key;
        } else {
            stream << request_key.substr(0, kMaxRequestKeyLength - 3) << "...";
        }
        const auto info = request_block_buffer->getDebugInfo();
        stream << "{rpc_completion=" << info.client_rpc_completion_count
               << ", callback_begin=" << info.client_callback_begin_count
               << ", callback_end=" << info.client_callback_end_count
               << ", callback_inflight=" << info.client_callback_inflight_count
               << ", last_rpc_completion_age_ms="
               << (info.last_client_rpc_completion_time_us > 0
                       ? (now_us - info.last_client_rpc_completion_time_us) / 1000
                       : -1)
               << ", last_callback_age_ms="
               << (info.last_client_callback_time_us > 0 ? (now_us - info.last_client_callback_time_us) / 1000 : -1)
               << "}";
    }
    stream << "], pending_layers=" << pending_count
           << ", omitted_pending_request_keys=" << pending_count - sampled_count << "}";
    return stream.str();
}

void SyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto                         once_time_ms = 30;
    while (true) {
        if (done_layer_cnt_ == expect_layer_cnt_) {
            return;
        }

        if (autil::TimeUtility::currentTimeInMilliSeconds() >= deadline_ms_) {
            auto error_code = ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT;
            error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
            RTP_LLM_LOG_WARNING("load context wait done on timeout, completion progress is %s",
                                getPendingRequestsDebugInfoLocked().c_str());
            return;
        }

        if (check_cancel_func_ != nullptr && check_cancel_func_()) {
            auto error_code = ErrorCode::CANCELLED;
            error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
            RTP_LLM_LOG_INFO("load context wait done on cancelled");
            return;
        }

        // sync wait, safe to use this
        if (cond_.wait_for(lock, std::chrono::milliseconds(once_time_ms), [this] {
                return done_layer_cnt_ == expect_layer_cnt_;
            })) {
            return;
        }
    }
}

bool SyncContext::success() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_.ok();
}

std::string SyncContext::getErrorInfoString() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_.ToString();
}

const ErrorInfo& SyncContext::getErrorInfo() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_;
}

LoadContext::LoadContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load):
    SyncContext(cache_store, combine_load) {}

void LoadContext::load(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffer,
                       const std::string&                                      ip,
                       uint32_t                                                port,
                       uint32_t                                                rdma_port,
                       int64_t                                                 timeout_ms,
                       CheckCancelFunc                                         check_cancel_func,
                       int                                                     partition_count,
                       int                                                     partition_id) {
    peer_ip_         = ip;
    port_            = port;
    rdma_port_       = rdma_port;
    partition_count_ = partition_count;
    partition_id_    = partition_id;
    call(request_block_buffer, timeout_ms, check_cancel_func);
}

bool LoadContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) {
    auto cache_store = cache_store_.lock();

    auto load_layer_callback = [request_block_buffer, shared_this = shared_from_this()](bool                success,
                                                                                        CacheStoreErrorCode ec) {
        shared_this->updateResult(success, ec, request_block_buffer);
    };
    cache_store->load(request_block_buffer,
                      load_layer_callback,
                      peer_ip_,
                      port_,
                      rdma_port_,
                      timeout_ms,
                      partition_count_,
                      partition_id_);
    return true;
}

StoreContext::StoreContext(const std::shared_ptr<CacheStore>& cache_store): SyncContext(cache_store, true) {}

void StoreContext::store(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                         int64_t                                                 timeout_ms) {
    call(request_block_buffers, timeout_ms, nullptr);
}

bool StoreContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) {
    auto cache_store = cache_store_.lock();

    auto store_layer_callback = [request_block_buffer, shared_this = shared_from_this()](bool                success,
                                                                                         CacheStoreErrorCode ec) {
        shared_this->updateResult(success, ec, request_block_buffer);
    };
    cache_store->store(request_block_buffer, store_layer_callback);
    return true;
}

}  // namespace rtp_llm
