#include <mutex>
#include <unordered_map>
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

RequestBlockBuffer::RequestBlockBuffer(const std::string& requestid, const std::string& request_key):
    requestid_(requestid), request_key_(request_key) {}

RequestBlockBuffer::RequestBlockBuffer(const std::string& requestid, std::shared_ptr<torch::Event> event):
    requestid_(requestid), event_(std::move(event)) {}

RequestBlockBuffer::~RequestBlockBuffer() {}

void RequestBlockBuffer::notifyRequestDone() {
    // request block buffer 关联的request已经结束，触发所有回调
    triggerWatchFunc(false, {});
}

void RequestBlockBuffer::markClientRpcCompletion() {
    ++client_rpc_completion_count_;
    last_client_rpc_completion_time_us_ = currentTimeUs();
}

void RequestBlockBuffer::markClientCallbackBegin() {
    ++client_callback_begin_count_;
    ++client_callback_inflight_count_;
    last_client_callback_time_us_ = currentTimeUs();
}

void RequestBlockBuffer::markClientCallbackEnd() {
    --client_callback_inflight_count_;
    ++client_callback_end_count_;
    last_client_callback_time_us_ = currentTimeUs();
}

const std::string& RequestBlockBuffer::getRequestId() const {
    return requestid_;
}

const std::string& RequestBlockBuffer::getRequestKey() const {
    return request_key_.empty() ? requestid_ : request_key_;
}

const torch::Event* RequestBlockBuffer::getEvent() const {
    return event_.get();
}

std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> RequestBlockBuffer::getBlocks() const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
    return blocks_;
}

std::shared_ptr<BlockBuffer> RequestBlockBuffer::getBlock(const std::string& id) const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);

    auto iter = blocks_.find(id);
    if (iter != blocks_.end()) {
        return iter->second;
    }
    return nullptr;
}

size_t RequestBlockBuffer::getBlocksCount() const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
    return blocks_.size();
}

size_t RequestBlockBuffer::getBlocksSize() const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
    return blocks_size_;
}

void RequestBlockBuffer::addBlock(const std::shared_ptr<BlockBuffer>& block) {
    if (block == nullptr) {
        return;
    }

    {
        std::unique_lock<std::shared_mutex> lock(blocks_mutex_);
        blocks_[block->key] = block;
        blocks_size_ += block->len;
    }
    ++block_add_count_;
    last_block_add_time_us_ = currentTimeUs();
    triggerWatchFunc(true, {block});
}

void RequestBlockBuffer::addBlock(
    const std::string& key, const std::shared_ptr<void>& addr, uint32_t len, bool gpu_mem, bool adopted) {
    auto block = std::make_shared<BlockBuffer>(key, addr, len, gpu_mem, adopted);
    addBlock(block);
}

void RequestBlockBuffer::addBlocks(const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    {
        std::unique_lock<std::shared_mutex> lock(blocks_mutex_);
        for (auto& block : blocks) {
            blocks_[block->key] = block;
            blocks_size_ += block->len;
        }
    }
    block_add_count_.fetch_add(blocks.size());
    last_block_add_time_us_ = currentTimeUs();

    triggerWatchFunc(true, blocks);
}

bool RequestBlockBuffer::isValid() const {
    std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
    for (auto iter : blocks_) {
        if (iter.second->addr == nullptr || iter.second->len == 0) {
            return false;
        }
    }
    return true;
}

bool RequestBlockBuffer::setWatchFunc(RequestBlockBuffer::WatchFunc&& watch_func) {
    // set callback
    {
        std::unique_lock<std::shared_mutex> lock(watch_func_mutex_);
        watch_funcs_.push_back(watch_func);
    }
    last_watch_register_time_us_ = currentTimeUs();

    // current blocks trigger once
    // set callback then trigger will not miss new blocks
    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    {
        std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
        for (auto iter : blocks_) {
            blocks.push_back(iter.second);
        }
    }
    if (!blocks.empty()) {
        triggerWatchFunc(true, blocks);
    }
    return true;
}

void RequestBlockBuffer::triggerWatchFunc(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    ++watch_trigger_count_;
    last_watch_trigger_time_us_ = currentTimeUs();
    std::vector<WatchFunc> tmp_watch_funcs;
    {
        std::shared_lock<std::shared_mutex> lock(watch_func_mutex_);
        tmp_watch_funcs = watch_funcs_;
    }

    for (auto watch_func : tmp_watch_funcs) {
        if (watch_func) {
            ++watch_callback_dispatch_count_;
            ++watch_callback_inflight_count_;
            watch_func(ok, blocks);
            --watch_callback_inflight_count_;
            ++watch_callback_complete_count_;
            last_watch_callback_complete_time_us_ = currentTimeUs();
        }
    }
}

RequestBlockBufferDebugInfo RequestBlockBuffer::getDebugInfo() const {
    RequestBlockBufferDebugInfo info;
    {
        std::shared_lock<std::shared_mutex> lock(blocks_mutex_);
        info.block_keys.reserve(blocks_.size());
        for (const auto& entry : blocks_) {
            info.block_keys.push_back(entry.first);
        }
    }
    {
        std::shared_lock<std::shared_mutex> lock(watch_func_mutex_);
        info.watch_func_count = watch_funcs_.size();
    }
    info.block_add_count                          = block_add_count_.load();
    info.watch_trigger_count                     = watch_trigger_count_.load();
    info.watch_callback_dispatch_count           = watch_callback_dispatch_count_.load();
    info.watch_callback_complete_count           = watch_callback_complete_count_.load();
    info.watch_callback_inflight_count           = watch_callback_inflight_count_.load();
    info.client_rpc_completion_count             = client_rpc_completion_count_.load();
    info.client_callback_begin_count              = client_callback_begin_count_.load();
    info.client_callback_end_count                = client_callback_end_count_.load();
    info.client_callback_inflight_count           = client_callback_inflight_count_.load();
    info.last_block_add_time_us                   = last_block_add_time_us_.load();
    info.last_watch_register_time_us              = last_watch_register_time_us_.load();
    info.last_watch_trigger_time_us               = last_watch_trigger_time_us_.load();
    info.last_watch_callback_complete_time_us     = last_watch_callback_complete_time_us_.load();
    info.last_client_rpc_completion_time_us       = last_client_rpc_completion_time_us_.load();
    info.last_client_callback_time_us             = last_client_callback_time_us_.load();
    return info;
}

std::string RequestBlockBuffer::debugInfo() const {
    const auto info = getDebugInfo();
    std::ostringstream stream;
    stream << "request id: " << requestid_ << ", blocks count: " << info.block_keys.size()
           << ", block adds: " << info.block_add_count << ", watch funcs: " << info.watch_func_count
           << ", watch triggers: " << info.watch_trigger_count
           << ", watch callback dispatches: " << info.watch_callback_dispatch_count
           << ", watch callback completes: " << info.watch_callback_complete_count
           << ", watch callback inflight: " << info.watch_callback_inflight_count;
    return stream.str();
}

}  // namespace rtp_llm
