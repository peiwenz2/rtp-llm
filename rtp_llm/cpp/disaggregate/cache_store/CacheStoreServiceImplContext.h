#pragma once
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

enum class CacheLoadFailureSource {
    UNKNOWN,
    TIMER_EXPIRED,
    REQUEST_BLOCK_BUFFER_CLOSED,
    WATCH_REGISTRATION_FAILED,
    INVALID_BLOCK,
    RESPONSE_WRITE_FAILED,
};

class CacheStoreServiceImplContext: public std::enable_shared_from_this<CacheStoreServiceImplContext> {
public:
    CacheStoreServiceImplContext(const CacheLoadRequest*                                      request,
                                 CacheLoadResponse*                                           response,
                                 const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                                 ::google::protobuf::Closure*                                 done,
                                 const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store);
    virtual ~CacheStoreServiceImplContext() = default;

public:
    void setTimer(const std::shared_ptr<Timer>& timer) {
        timer_ = std::weak_ptr<Timer>(timer);
    }
    void runFailed(KvCacheStoreServiceErrorCode error_code,
                   CacheLoadFailureSource       failure_source = CacheLoadFailureSource::UNKNOWN);
    void markWatchRegisterAttempt();
    void markWatchRegisterResult(bool success);
    void markWatchCallbackBegin(size_t block_count);
    void markWatchCallbackEnd();

protected:
    std::shared_ptr<BlockBufferInfo> getAndEraseUnLoadedBlock(const std::string& block_key);
    std::string                      getLoadProgressDebugInfo();
    std::string getLoadProgressDebugInfo(const std::optional<RequestBlockBufferDebugInfo>& block_buffer_debug_info);
    void                             stopTimer();
    void                             runSuccess(bool direct_write);

protected:
    const CacheLoadRequest* request_;
    const int64_t           context_create_time_us_{0};
    const int64_t           request_send_start_time_us_{0};
    const uint32_t          total_block_count_{0};
    uint32_t                unique_block_count_{0};
    const std::string       request_id_;
    const std::string       peer_ip_;
    const int32_t           partition_count_{1};
    const int32_t           partition_id_{0};

    std::mutex         response_mutex_;
    CacheLoadResponse* response_;

    std::shared_ptr<CacheStoreServerLoadMetricsCollector> collector_;

    std::atomic_bool             done_run_{false};
    ::google::protobuf::Closure* done_;
    std::atomic<CacheLoadFailureSource> failure_source_{CacheLoadFailureSource::UNKNOWN};

    std::weak_ptr<RequestBlockBufferStore> request_block_buffer_store_;

    std::weak_ptr<Timer> timer_;

    std::atomic_int write_cnt_{0};

    std::atomic_int     watch_register_result_{-1};
    std::atomic_uint64_t watch_callback_enter_count_{0};
    std::atomic_uint64_t watch_callback_exit_count_{0};
    std::atomic_uint64_t watch_callback_inflight_count_{0};
    std::atomic_uint64_t watch_callback_block_count_{0};
    std::atomic_int64_t  watch_register_time_us_{0};
    std::atomic_int64_t  last_watch_callback_time_us_{0};
    std::atomic_int64_t  last_block_match_time_us_{0};
    std::atomic_int64_t  last_response_write_time_us_{0};

    std::shared_mutex                                                 unloaded_blocks_mutex_;
    std::unordered_map<std::string, std::shared_ptr<BlockBufferInfo>> unloaded_blocks_;
    std::vector<std::string>                                          duplicate_block_key_samples_;
};
}  // namespace rtp_llm
