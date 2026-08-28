#pragma once

#include <torch/all.h>

#include <shared_mutex>
#include <unordered_map>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace rtp_llm {

struct RequestBlockBufferDebugInfo {
    std::vector<std::string> block_keys;
    size_t                   block_add_count{0};
    size_t                   watch_func_count{0};
    uint64_t                 watch_trigger_count{0};
    uint64_t                 watch_callback_dispatch_count{0};
    uint64_t                 watch_callback_complete_count{0};
    uint64_t                 watch_callback_inflight_count{0};
    uint64_t                 client_rpc_completion_count{0};
    uint64_t                 client_callback_begin_count{0};
    uint64_t                 client_callback_end_count{0};
    uint64_t                 client_callback_inflight_count{0};
    int64_t                  last_block_add_time_us{0};
    int64_t                  last_watch_register_time_us{0};
    int64_t                  last_watch_trigger_time_us{0};
    int64_t                  last_watch_callback_complete_time_us{0};
    int64_t                  last_client_rpc_completion_time_us{0};
    int64_t                  last_client_callback_time_us{0};
};

// 关联一块内存/显存
class BlockBuffer {
public:
    BlockBuffer(
        const std::string& key_, const std::shared_ptr<void>& addr_, uint32_t len_, bool gpu_mem_, bool adopted_):
        key(key_), addr(addr_), len(len_), gpu_mem(gpu_mem_), adopted(adopted_) {}
    BlockBuffer(const BlockBuffer& rhs):
        key(rhs.key), addr(rhs.addr), len(rhs.len), gpu_mem(rhs.gpu_mem), adopted(rhs.adopted) {}

    std::string           key;
    std::shared_ptr<void> addr;
    uint32_t              len{0};
    bool                  gpu_mem{true};
    bool                  adopted{true};
};

//  request 关联的 block buffer
class RequestBlockBuffer {
public:
    RequestBlockBuffer(const std::string& requestid, const std::string& request_key = "");
    RequestBlockBuffer(const std::string& requestid, std::shared_ptr<torch::Event> event);

    ~RequestBlockBuffer();

public:
    const std::string&  getRequestId() const;
    const std::string&  getRequestKey() const;
    const torch::Event* getEvent() const;

    std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> getBlocks() const;
    std::shared_ptr<BlockBuffer>                                  getBlock(const std::string& id) const;
    size_t                                                        getBlocksCount() const;
    size_t                                                        getBlocksSize() const;

    void addBlock(const std::shared_ptr<BlockBuffer>& block);
    void addBlock(const std::string& key, const std::shared_ptr<void>& addr, uint32_t len, bool gpu_mem, bool adopted);
    void addBlocks(const std::vector<std::shared_ptr<BlockBuffer>>& blocks);

    bool isValid() const;

    // change with true callback, dtor with false callback
    typedef std::function<void(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&)> WatchFunc;
    bool setWatchFunc(WatchFunc&& watch_func);
    void notifyRequestDone();
    void markClientRpcCompletion();
    void markClientCallbackBegin();
    void markClientCallbackEnd();

    RequestBlockBufferDebugInfo getDebugInfo() const;
    std::string                 debugInfo() const;

private:
    void triggerWatchFunc(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>&);

private:
    std::string requestid_;
    std::string request_key_;

    std::shared_ptr<torch::Event> event_;

    mutable std::shared_mutex                                     blocks_mutex_;
    std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> blocks_;
    size_t                                                        blocks_size_ = 0;

    mutable std::shared_mutex watch_func_mutex_;
    std::vector<WatchFunc>    watch_funcs_;

    std::atomic_size_t   block_add_count_{0};
    std::atomic_uint64_t watch_trigger_count_{0};
    std::atomic_uint64_t watch_callback_dispatch_count_{0};
    std::atomic_uint64_t watch_callback_complete_count_{0};
    std::atomic_uint64_t watch_callback_inflight_count_{0};
    std::atomic_uint64_t client_rpc_completion_count_{0};
    std::atomic_uint64_t client_callback_begin_count_{0};
    std::atomic_uint64_t client_callback_end_count_{0};
    std::atomic_uint64_t client_callback_inflight_count_{0};
    std::atomic_int64_t  last_block_add_time_us_{0};
    std::atomic_int64_t  last_watch_register_time_us_{0};
    std::atomic_int64_t  last_watch_trigger_time_us_{0};
    std::atomic_int64_t  last_watch_callback_complete_time_us_{0};
    std::atomic_int64_t  last_client_rpc_completion_time_us_{0};
    std::atomic_int64_t  last_client_callback_time_us_{0};
};

}  // namespace rtp_llm
