#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"

namespace rtp_llm {

class RequestBlockBufferTest: public ::testing::Test {};

TEST_F(RequestBlockBufferTest, testBlockOps) {

    RequestBlockBuffer buffer("test-request-id");

    ASSERT_EQ(0, buffer.getBlocksCount());

    std::shared_ptr<void> buffer1((void*)0x1, [](void* p) {});
    buffer.addBlock(std::make_shared<BlockBuffer>("b1", buffer1, 10, true, true));
    ASSERT_EQ(1, buffer.getBlocksCount());
    ASSERT_TRUE(buffer.isValid());
    ASSERT_EQ(buffer1, buffer.getBlock("b1")->addr);
    ASSERT_EQ(10, buffer.getBlocksSize());

    std::shared_ptr<void> buffer2((void*)0x2, [](void* p) {});
    buffer.addBlock("b2", buffer2, 10, true, true);
    ASSERT_EQ(2, buffer.getBlocksCount());
    ASSERT_TRUE(buffer.isValid());
    ASSERT_EQ(buffer2, buffer.getBlock("b2")->addr);
    ASSERT_EQ(20, buffer.getBlocksSize());

    buffer.addBlock("b3", nullptr, 10, true, true);
    ASSERT_EQ(3, buffer.getBlocksCount());
    ASSERT_FALSE(buffer.isValid());
    ASSERT_EQ(nullptr, buffer.getBlock("b3")->addr);
    ASSERT_EQ(30, buffer.getBlocksSize());
}

TEST_F(RequestBlockBufferTest, testWatchFunc_SetWatchFunc) {
    {
        // set to empty request block buffer
        bool                                      watched_called{false};
        bool                                      watched_success{false};
        std::vector<std::shared_ptr<BlockBuffer>> watched_blocks;
        auto                                      watch_func = [&watched_called, &watched_success, &watched_blocks](
                              bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            watched_called  = true;
            watched_success = success;
            watched_blocks  = blocks;
        };

        auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
        ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func)));
        ASSERT_FALSE(watched_called);
        ASSERT_FALSE(watched_success);
        ASSERT_TRUE(watched_blocks.empty());

        // set twice
        watched_called   = false;
        watched_success  = false;
        auto watch_func2 = [&watched_called, &watched_success, &watched_blocks](
                               bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            watched_called  = true;
            watched_success = success;
            watched_blocks  = blocks;
        };
        ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func2)));
        ASSERT_FALSE(watched_called);
        ASSERT_FALSE(watched_success);
        ASSERT_TRUE(watched_blocks.empty());

        request_block_buffer.reset();
    }
    {
        // set to non empty request block buffer
        auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
        request_block_buffer->addBlock(std::make_shared<BlockBuffer>("b1", nullptr, 10, true, true));
        request_block_buffer->addBlock(std::make_shared<BlockBuffer>("b2", nullptr, 10, false, false));

        bool                                      watched_called1{false};
        bool                                      watched_success1{false};
        std::vector<std::shared_ptr<BlockBuffer>> watched_blocks;
        auto                                      watch_func1 = [&watched_called1, &watched_success1, &watched_blocks](
                               bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            watched_called1  = true;
            watched_success1 = success;
            watched_blocks.insert(watched_blocks.end(), blocks.begin(), blocks.end());
        };

        ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func1)));
        ASSERT_TRUE(watched_called1);
        ASSERT_TRUE(watched_success1);
        ASSERT_EQ(2, watched_blocks.size());

        // set twice
        watched_called1  = false;
        watched_success1 = false;
        watched_blocks.clear();
        bool watched_called2{false};
        bool watched_success2{false};
        auto watch_func2 = [&watched_called2, &watched_success2, &watched_blocks](
                               bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            watched_called2  = true;
            watched_success2 = success;
            watched_blocks.insert(watched_blocks.end(), blocks.begin(), blocks.end());
        };
        ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func2)));
        ASSERT_TRUE(watched_called1);
        ASSERT_TRUE(watched_success1);
        ASSERT_TRUE(watched_called2);
        ASSERT_TRUE(watched_success2);
        ASSERT_EQ(4, watched_blocks.size());

        request_block_buffer.reset();
    }
}

TEST_F(RequestBlockBufferTest, testWatchFunc_DebugInfoTracksDispatchLifecycle) {
    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    int  callback_count       = 0;
    ASSERT_TRUE(request_block_buffer->setWatchFunc(
        [&callback_count](bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
            ASSERT_TRUE(ok);
            ASSERT_EQ(blocks.size(), 1);
            ++callback_count;
        }));

    std::shared_ptr<void> buffer((void*)0x1, [](void*) {});
    request_block_buffer->addBlock(std::make_shared<BlockBuffer>("b0", buffer, 1024, true, true));

    const auto info = request_block_buffer->getDebugInfo();
    EXPECT_EQ(callback_count, 1);
    EXPECT_EQ(info.block_keys.size(), 1);
    EXPECT_EQ(info.block_add_count, 1);
    EXPECT_EQ(info.watch_func_count, 1);
    EXPECT_EQ(info.watch_trigger_count, 1);
    EXPECT_EQ(info.watch_callback_dispatch_count, 1);
    EXPECT_EQ(info.watch_callback_complete_count, 1);
    EXPECT_EQ(info.watch_callback_inflight_count, 0);
    EXPECT_GT(info.last_block_add_time_us, 0);
    EXPECT_GT(info.last_watch_register_time_us, 0);
    EXPECT_GT(info.last_watch_trigger_time_us, 0);
    EXPECT_GT(info.last_watch_callback_complete_time_us, 0);

    request_block_buffer->markClientRpcCompletion();
    request_block_buffer->markClientCallbackBegin();
    auto inflight_info = request_block_buffer->getDebugInfo();
    EXPECT_EQ(inflight_info.client_rpc_completion_count, 1);
    EXPECT_EQ(inflight_info.client_callback_begin_count, 1);
    EXPECT_EQ(inflight_info.client_callback_end_count, 0);
    EXPECT_EQ(inflight_info.client_callback_inflight_count, 1);
    request_block_buffer->markClientCallbackEnd();
    const auto completed_info = request_block_buffer->getDebugInfo();
    EXPECT_EQ(completed_info.client_callback_end_count, 1);
    EXPECT_EQ(completed_info.client_callback_inflight_count, 0);
    EXPECT_GT(completed_info.last_client_rpc_completion_time_us, 0);
    EXPECT_GT(completed_info.last_client_callback_time_us, 0);
}

TEST_F(RequestBlockBufferTest, testWatchFunc_AddBlock) {
    bool                                      watched_called1{false};
    bool                                      watched_success1{false};
    std::vector<std::shared_ptr<BlockBuffer>> watched_blocks;
    auto                                      watch_func1 = [&watched_called1, &watched_success1, &watched_blocks](
                           bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
        watched_called1  = true;
        watched_success1 = success;
        watched_blocks.insert(watched_blocks.end(), blocks.begin(), blocks.end());
    };

    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func1)));

    // add block
    request_block_buffer->addBlock(std::make_shared<BlockBuffer>("b1", nullptr, 10, true, true));
    ASSERT_TRUE(watched_called1);
    ASSERT_TRUE(watched_success1);
    ASSERT_EQ(1, watched_blocks.size());
    ASSERT_EQ("b1", watched_blocks[0]->key);

    watched_called1 = false;
    watched_blocks.clear();
    bool watched_called2{false};
    bool watched_success2{false};
    auto watch_func2 = [&watched_called2, &watched_success2, &watched_blocks](
                           bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
        watched_called2  = true;
        watched_success2 = success;
        watched_blocks.insert(watched_blocks.end(), blocks.begin(), blocks.end());
    };
    ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func2)));
    ASSERT_EQ(2, watched_blocks.size());
    watched_called1 = false;
    watched_called2 = false;
    watched_blocks.clear();

    // add block
    request_block_buffer->addBlock("b2", nullptr, 10, true, true);
    ASSERT_TRUE(watched_called1);
    ASSERT_TRUE(watched_success1);
    ASSERT_TRUE(watched_called2);
    ASSERT_TRUE(watched_success2);
    ASSERT_EQ(2, watched_blocks.size());
    ASSERT_EQ("b2", watched_blocks[1]->key);

    // add blocks
    watched_called1 = false;
    watched_called2 = false;
    watched_blocks.clear();
    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    blocks.push_back(std::make_shared<BlockBuffer>("b3", nullptr, 10, true, true));
    blocks.push_back(std::make_shared<BlockBuffer>("b4", nullptr, 10, true, true));
    request_block_buffer->addBlocks(blocks);
    ASSERT_TRUE(watched_called1);
    ASSERT_TRUE(watched_success1);
    ASSERT_TRUE(watched_called2);
    ASSERT_TRUE(watched_success2);
    ASSERT_EQ(4, watched_blocks.size());
    ASSERT_EQ("b3", watched_blocks[2]->key);
    ASSERT_EQ("b4", watched_blocks[3]->key);

    request_block_buffer.reset();
}

TEST_F(RequestBlockBufferTest, testWatchFunc_ReleaseBlock) {
    bool                                      watched_called{false};
    bool                                      watched_success{false};
    std::vector<std::shared_ptr<BlockBuffer>> watched_blocks;
    auto                                      watch_func = [&watched_called, &watched_success, &watched_blocks](
                          bool success, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
        watched_called  = true;
        watched_success = success;
        watched_blocks  = blocks;
    };

    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    ASSERT_TRUE(request_block_buffer->setWatchFunc(std::move(watch_func)));

    request_block_buffer->notifyRequestDone();
    ASSERT_TRUE(watched_called);
    ASSERT_FALSE(watched_success);
    ASSERT_TRUE(watched_blocks.empty());
}

}  // namespace rtp_llm
