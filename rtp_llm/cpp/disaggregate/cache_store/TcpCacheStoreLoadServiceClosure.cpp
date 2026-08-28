#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include <torch/torch.h>
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

TcpCacheStoreLoadServiceClosure::~TcpCacheStoreLoadServiceClosure() {
    if (controller_) {
        delete controller_;
    }
    if (request_) {
        delete request_;
    }
    if (response_) {
        delete response_;
    }
}

void TcpCacheStoreLoadServiceClosure::Run() {
    pinThreadToDeviceOnce(device_id_);
    collector_->markRequestCallEnd(currentTimeUs() - response_->response_send_start_time_us());
    const auto request_id = request_ != nullptr ? request_->requestid() : "unknown";
    request_block_buffer_->markClientRpcCompletion();

    if (controller_->Failed()) {
        RTP_LLM_LOG_WARNING("cache load rpc completion received with controller failure, request %s, controller err %d",
                            request_id.c_str(),
                            controller_->GetErrorCode());
        end(false, CacheStoreUtil::fromArpcErrorCode(controller_->GetErrorCode()));
        return;
    }

    if (response_->error_code() != KvCacheStoreServiceErrorCode::EC_SUCCESS) {
        RTP_LLM_LOG_WARNING("cache load rpc completion received with response failure, request %s, response err %d",
                            request_id.c_str(),
                            response_->error_code());
        end(false, CacheStoreUtil::fromKvCacheStoreErrorCode(response_->error_code()));
        return;
    }

    // TCP Mode 下需要Copy数据
    if (response_->blocks_size() != request_block_buffer_->getBlocksCount()) {
        RTP_LLM_LOG_WARNING("cache load response block count not equal to request block buffer");
        end(false, CacheStoreErrorCode::LoadBufferTimeout);
        return;
    }

    for (int i = 0; i < response_->blocks_size(); i++) {
        const auto& block        = response_->blocks(i);
        auto        unload_block = request_block_buffer_->getBlock(block.key());

        if (unload_block == nullptr || block.len() != unload_block->len) {
            RTP_LLM_LOG_WARNING("can not find match block %s from response, request is %s",
                                block.key().c_str(),
                                request_block_buffer_->getRequestId().c_str());
            end(false, CacheStoreErrorCode::LoadBufferTimeout);
            return;
        }

        auto dst_tensor = torch::from_blob(
            unload_block->addr.get(),
            {(int64_t)unload_block->len},
            torch::TensorOptions().dtype(torch::kUInt8).device(unload_block->gpu_mem ? torch::kCUDA : torch::kCPU));
        auto src_tensor = torch::from_blob(const_cast<char*>(block.content().data()),
                                           {(int64_t)block.len()},
                                           torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        execNoBlockCopy({dst_tensor, src_tensor});
    }
    end(true, CacheStoreErrorCode::None);
}

void TcpCacheStoreLoadServiceClosure::end(bool success, CacheStoreErrorCode ec) {
    collector_->markEnd(success);
    const auto request_id = request_ != nullptr ? request_->requestid() : "unknown";
    if (!success) {
        RTP_LLM_LOG_WARNING("cache load client callback begin, request %s, error %s",
                            request_id.c_str(),
                            CacheStoreErrorCodeToString(ec).c_str());
    }
    request_block_buffer_->markClientCallbackBegin();
    callback_(success, ec);
    request_block_buffer_->markClientCallbackEnd();
    if (!success) {
        RTP_LLM_LOG_WARNING("cache load client callback end, request %s, error %s",
                            request_id.c_str(),
                            CacheStoreErrorCodeToString(ec).c_str());
    }
    delete this;
}

}  // namespace rtp_llm
