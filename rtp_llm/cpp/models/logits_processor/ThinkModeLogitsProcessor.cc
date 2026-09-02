#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include <algorithm>
#include <limits>

using namespace std;

namespace rtp_llm {

namespace {

constexpr int32_t kInvalidTokenId = -1;

void maskToken(const torch::Tensor& new_tokens_logits, size_t vocab_size, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
        return;
    }
    new_tokens_logits[token_id] = BaseLogitsProcessor::neg_inf;
}

int generatedTokens(const SamplerInputs& inputs, size_t batch_idx) {
    int* input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
    int* sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();
    return sequence_lengths[batch_idx] - input_lengths[batch_idx];
}

bool thinkBudgetExhausted(const SamplerInputs& inputs, size_t batch_idx, const StreamThinkInfo& info) {
    if (!info.dfa_ptr || info.end_think_token_ids.empty() || info.max_thinking_tokens <= 0) {
        return false;
    }

    const int observed_output_tokens = std::max(generatedTokens(inputs, batch_idx), info.current_output_length);
    return observed_output_tokens >= info.max_thinking_tokens;
}

void maskThinkBoundaryTokens(const torch::Tensor& new_tokens_logits, size_t vocab_size, const StreamThinkInfo& info) {
    maskToken(new_tokens_logits, vocab_size, info.beginBoundaryTokenToMask());
    maskToken(new_tokens_logits, vocab_size, info.endBoundaryTokenToMask());
}

void maskReasoningStopTokens(const torch::Tensor& new_tokens_logits,
                             size_t                vocab_size,
                             const StreamThinkInfo& info) {
    for (const int token_id : info.reasoning_stop_token_ids) {
        maskToken(new_tokens_logits, vocab_size, token_id);
    }
}

void clearTokenFromBitmask(int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    row[token_id / 32] &= ~(1u << (token_id % 32));
}

void forceTokenInBitmask(int32_t* row, size_t words, int32_t token_id) {
    std::fill_n(row, words, 0);
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    row[token_id / 32] |= (1u << (token_id % 32));
}

bool bitmaskAllowsToken(const int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return false;
    }
    const uint32_t word = static_cast<uint32_t>(row[token_id / 32]);
    return (word & (1u << (token_id % 32))) != 0u;
}

bool specThinkBudgetExhausted(const StreamThinkInfo& info) {
    return info.dfa_ptr && !info.end_think_token_ids.empty() && info.max_thinking_tokens > 0
           && info.current_output_length >= info.max_thinking_tokens;
}

bool forceThinkEndTokenInBitmask(int32_t* row, size_t words, const StreamThinkInfo& info) {
    if (!info.dfa_ptr || info.dfa_ptr->isFinished() || info.end_think_token_ids.empty()) {
        return false;
    }
    auto next_token_idx = info.dfa_ptr->status();
    if (next_token_idx >= info.end_think_token_ids.size()) {
        return false;
    }
    forceTokenInBitmask(row, words, info.end_think_token_ids[next_token_idx]);
    return true;
}

void applyThinkSpecRowMask(int32_t* row, size_t words, StreamThinkInfo& info) {
    std::fill_n(row, words, SpecLogitsProcessor::kBitmaskAllowAll);
    if (!info.pending_forced_think_end_token_ids.empty()) {
        forceTokenInBitmask(row, words, info.pending_forced_think_end_token_ids.front());
        return;
    }
    switch (info.process_state) {
        case ThinkProcessState::NO_THINK:
        case ThinkProcessState::AFTER_THINK: {
            clearTokenFromBitmask(row, words, info.beginBoundaryTokenToMask());
            clearTokenFromBitmask(row, words, info.endBoundaryTokenToMask());
            break;
        }
        case ThinkProcessState::IN_THINK: {
            if (info.transitionToAfterThinkIfClosed()) {
                clearTokenFromBitmask(row, words, info.beginBoundaryTokenToMask());
                clearTokenFromBitmask(row, words, info.endBoundaryTokenToMask());
                break;
            }
            if (info.closeInProgress() || specThinkBudgetExhausted(info)) {
                info.process_state = ThinkProcessState::CLOSING_THINK;
                if (!forceThinkEndTokenInBitmask(row, words, info)) {
                    clearTokenFromBitmask(row, words, info.beginBoundaryTokenToMask());
                    clearTokenFromBitmask(row, words, info.endBoundaryTokenToMask());
                    for (const int token_id : info.reasoning_stop_token_ids) {
                        clearTokenFromBitmask(row, words, token_id);
                    }
                }
                break;
            }
            clearTokenFromBitmask(row, words, info.beginBoundaryTokenToMask());
            for (const int token_id : info.reasoning_stop_token_ids) {
                clearTokenFromBitmask(row, words, token_id);
            }
            break;
        }
        case ThinkProcessState::CLOSING_THINK: {
            if (info.transitionToAfterThinkIfClosed()) {
                clearTokenFromBitmask(row, words, info.beginBoundaryTokenToMask());
                clearTokenFromBitmask(row, words, info.endBoundaryTokenToMask());
                break;
            }
            if (!forceThinkEndTokenInBitmask(row, words, info)) {
                clearTokenFromBitmask(row, words, info.beginBoundaryTokenToMask());
                clearTokenFromBitmask(row, words, info.endBoundaryTokenToMask());
                for (const int token_id : info.reasoning_stop_token_ids) {
                    clearTokenFromBitmask(row, words, token_id);
                }
            }
            break;
        }
    }
}

}  // namespace

bool StreamThinkInfo::isActive() const {
    return process_state == ThinkProcessState::IN_THINK || process_state == ThinkProcessState::CLOSING_THINK;
}

bool StreamThinkInfo::transitionToAfterThinkIfClosed() {
    if (!dfa_ptr || !dfa_ptr->isFinished()) {
        return false;
    }
    process_state = ThinkProcessState::AFTER_THINK;
    return true;
}

bool StreamThinkInfo::closeInProgress() const {
    return dfa_ptr && dfa_ptr->status() > 0;
}

int32_t StreamThinkInfo::tokenCompletingBoundary(const std::vector<int>& boundary) const {
    if (boundary.empty()) {
        return kInvalidTokenId;
    }
    const size_t prefix_size = boundary.size() - 1;
    if (boundary_history.size() < prefix_size
        || !std::equal(boundary.begin(), boundary.end() - 1, boundary_history.end() - prefix_size)) {
        return kInvalidTokenId;
    }
    return boundary.back();
}

int32_t StreamThinkInfo::beginBoundaryTokenToMask() const {
    return tokenCompletingBoundary(begin_think_token_ids);
}

int32_t StreamThinkInfo::endBoundaryTokenToMask() const {
    return tokenCompletingBoundary(end_think_token_ids);
}

void StreamThinkInfo::advanceBoundaryHistory(int32_t token_id) {
    const size_t max_boundary_size = std::max(begin_think_token_ids.size(), end_think_token_ids.size());
    if (max_boundary_size <= 1) {
        return;
    }
    boundary_history.push_back(token_id);
    if (boundary_history.size() >= max_boundary_size) {
        boundary_history.erase(boundary_history.begin());
    }
}

bool StreamThinkInfo::consumePendingForcedToken(int32_t token_id) {
    if (pending_forced_think_end_token_ids.empty()) {
        return false;
    }
    const int32_t expected_token_id = pending_forced_think_end_token_ids.front();
    pending_forced_think_end_token_ids.erase(pending_forced_think_end_token_ids.begin());
    if (token_id != expected_token_id) {
        RTP_LLM_LOG_WARNING("forced think end token mismatch, expected=%d actual=%d, trust precommitted state",
                            expected_token_id,
                            token_id);
    }
    return true;
}

void StreamThinkInfo::advanceToken(int32_t token_id) {
    if (consumePendingForcedToken(token_id)) {
        return;
    }
    current_output_length += 1;
    advanceBoundaryHistory(token_id);
    if (!isActive() || !dfa_ptr) {
        return;
    }
    dfa_ptr->next(token_id);
    if (dfa_ptr->isFinished()) {
        process_state = ThinkProcessState::AFTER_THINK;
    } else if (closeInProgress()) {
        process_state = ThinkProcessState::CLOSING_THINK;
    } else if (process_state == ThinkProcessState::CLOSING_THINK) {
        process_state = ThinkProcessState::IN_THINK;
    }
}

void StreamThinkInfo::precommitForcedToken(int32_t token_id) {
    dfa_ptr->next(token_id);
    pending_forced_think_end_token_ids.push_back(token_id);
    current_output_length += 1;
    advanceBoundaryHistory(token_id);
    process_state = dfa_ptr->isFinished() ? ThinkProcessState::AFTER_THINK : ThinkProcessState::CLOSING_THINK;
}

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(std::vector<StreamThinkInfo> think_infos):
    think_infos_(think_infos) {
    std::lock_guard<std::mutex> lock(mutex_);
    publishSpecSnapshotLocked();
};

void ThinkModeLogitsProcessor::publishSpecSnapshotLocked() {
    auto snapshot      = std::make_shared<ThinkModeSpecSnapshot>();
    snapshot->version  = ++spec_snapshot_version_;
    snapshot->eligible = think_infos_.size() == 1 && !think_infos_[0].is_beam_search;
    if (snapshot->eligible) {
        snapshot->info = think_infos_[0].copy();
    }
    std::atomic_store_explicit(
        &spec_snapshot_, std::shared_ptr<const ThinkModeSpecSnapshot>(snapshot), std::memory_order_release);
}

void ThinkModeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(think_infos_.size() == finish_idx - start_idx);

    for (size_t i = 0; i < think_infos_.size(); ++i) {
        auto&  info      = think_infos_[i];
        size_t batch_idx = i + start_idx;

        switch (info.process_state) {
            case ThinkProcessState::NO_THINK:
            case ThinkProcessState::AFTER_THINK: {
                maskThinkBoundaryTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                break;
            }
            case ThinkProcessState::IN_THINK: {
                if (info.transitionToAfterThinkIfClosed()) {
                    maskThinkBoundaryTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                    break;
                }

                const bool boundary_in_progress = info.closeInProgress();
                const bool budget_exhausted     = thinkBudgetExhausted(inputs, batch_idx, info);
                if (boundary_in_progress || budget_exhausted) {
                    info.process_state = ThinkProcessState::CLOSING_THINK;
                    forceThinkEndToken(inputs.logits[batch_idx],
                                       info,
                                       inputs.vocab_size,
                                       boundary_in_progress ? "boundary" : "budget");
                    break;
                }

                maskToken(inputs.logits[batch_idx], inputs.vocab_size, info.beginBoundaryTokenToMask());
                maskReasoningStopTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                break;
            }
            case ThinkProcessState::CLOSING_THINK: {
                if (info.transitionToAfterThinkIfClosed()) {
                    maskThinkBoundaryTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                    break;
                }

                if (!forceThinkEndToken(inputs.logits[batch_idx], info, inputs.vocab_size, "boundary")) {
                    maskThinkBoundaryTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                    maskReasoningStopTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                }
                break;
            }
        }
    }
    publishSpecSnapshotLocked();
}

bool ThinkModeLogitsProcessor::forceThinkEndToken(const torch::Tensor& new_tokens_logits,
                                                  StreamThinkInfo&     info,
                                                  size_t               vocab_size,
                                                  const char*          trigger) {
    if (!info.dfa_ptr || info.dfa_ptr->isFinished() || info.end_think_token_ids.empty()) {
        return false;
    }
    auto next_token_idx = info.dfa_ptr->status();
    if (next_token_idx >= info.end_think_token_ids.size()) {
        return false;
    }

    auto token_id = info.end_think_token_ids[next_token_idx];
    RTP_LLM_LOG_INFO("force think boundary: trigger=%s token=%d progress=%zu/%zu",
                     trigger,
                     token_id,
                     next_token_idx + 1,
                     info.end_think_token_ids.size());
    memFill(new_tokens_logits, vocab_size, (size_t)token_id);

    // Beam/multi-sequence updates need src-batch remapping from updateStatus(),
    // and they do not use the normal async device-state fast path. Keep their
    // historical behavior: force logits now, advance DFA when the sampled token
    // is committed by updateStatus().
    if (info.is_beam_search) {
        return true;
    }

    info.precommitForcedToken(token_id);
    return true;
}

void ThinkModeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::lock_guard<std::mutex>  lock(mutex_);
    std::vector<StreamThinkInfo> new_think_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_think_infos.push_back(think_infos_[src_batch_idx].copy());
    }
    think_infos_ = new_think_infos;
    publishSpecSnapshotLocked();
}

void ThinkModeLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(think_infos_.size() == (size_t)new_tokens.size(0));

    for (size_t i = 0; i < think_infos_.size(); i++) {
        auto& info = think_infos_[i];
        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK_WITH_INFO(num_new_tokens <= new_tokens.size(1),
                                    "think mode commit token count exceeds tensor width, num_new_tokens=%d, "
                                    "new_tokens.size(1)=%ld",
                                    num_new_tokens,
                                    new_tokens.size(1));
        }

        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = new_tokens.data_ptr<int>()[i * new_tokens.size(1) + j + offset];
            info.advanceToken(current_token_id);
        }
    }
    publishSpecSnapshotLocked();
}

bool ThinkModeLogitsProcessor::isSpecVerifyEligible() const {
    auto snapshot = std::atomic_load_explicit(&spec_snapshot_, std::memory_order_acquire);
    return snapshot && snapshot->eligible;
}

bool ThinkModeLogitsProcessor::isStateful() const {
    return isSpecVerifyEligible();
}

int64_t ThinkModeLogitsProcessor::acceptedTokenLen() const {
    auto snapshot = std::atomic_load_explicit(&spec_snapshot_, std::memory_order_acquire);
    if (!snapshot || !snapshot->eligible) {
        return 0;
    }
    return snapshot->info.current_output_length;
}

int ThinkModeLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    auto snapshot = std::atomic_load_explicit(&spec_snapshot_, std::memory_order_acquire);
    if (!snapshot || !snapshot->eligible || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return request.propose_step;
    }

    StreamThinkInfo state = snapshot->info.copy();
    int             cap   = request.propose_step;
    const size_t    W     = request.bitmask_size_int32;

    for (int offset = 0; offset <= request.propose_step; ++offset) {
        int32_t* row = request.bitmask_cpu_out + offset * W;
        applyThinkSpecRowMask(row, W, state);
        if (offset == request.propose_step) {
            break;
        }

        const int32_t draft_token = request.draft_tokens[offset];
        if (!bitmaskAllowsToken(row, W, draft_token)) {
            cap = offset;
            break;
        }
        state.advanceToken(draft_token);
    }
    return cap;
}

ThinkModeLogitsProcessorPtr ThinkModeLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                        int32_t                        num,
                                                                        int64_t                        eos_token_id) {
    auto generate_config         = generate_input->generate_config;
    auto end_think_token_ids     = generate_config->end_think_token_ids;
    bool has_think_boundary_mask = !generate_config->begin_think_token_ids.empty() || !end_think_token_ids.empty();
    if (!has_think_boundary_mask) {
        return nullptr;
    }

    std::vector<int> reasoning_stop_token_ids;
    if (generate_config->in_think_mode) {
        for (const auto& stop_word : generate_config->stop_words_list) {
            if (stop_word.size() == 1) {
                reasoning_stop_token_ids.push_back(stop_word.front());
            }
        }
        if (eos_token_id >= 0 && eos_token_id <= std::numeric_limits<int>::max()) {
            reasoning_stop_token_ids.push_back(static_cast<int>(eos_token_id));
        }
        if (!end_think_token_ids.empty()) {
            reasoning_stop_token_ids.erase(
                std::remove(reasoning_stop_token_ids.begin(),
                            reasoning_stop_token_ids.end(),
                            end_think_token_ids.front()),
                reasoning_stop_token_ids.end());
        }
        std::sort(reasoning_stop_token_ids.begin(), reasoning_stop_token_ids.end());
        reasoning_stop_token_ids.erase(
            std::unique(reasoning_stop_token_ids.begin(), reasoning_stop_token_ids.end()),
            reasoning_stop_token_ids.end());
    }

    std::vector<StreamThinkInfo> think_infos;
    think_infos.reserve(num);
    for (size_t i = 0; i < num; i++) {
        std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr;
        if (generate_config->in_think_mode && !end_think_token_ids.empty()) {
            dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(end_think_token_ids);
        }
        think_infos.emplace_back(generate_config->in_think_mode,
                                 generate_config->max_thinking_tokens,
                                 generate_config->begin_think_token_ids,
                                 end_think_token_ids,
                                 generate_input->inputLength(),
                                 0,
                                 generate_config->hasNumBeams() || generate_config->num_return_sequences > 1,
                                 std::move(dfa_ptr),
                                 reasoning_stop_token_ids);
    }
    return std::make_shared<ThinkModeLogitsProcessor>(std::move(think_infos));
}

std::vector<size_t> ThinkModeLogitsProcessor::thinkEndTokensStatus() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<size_t>         status;
    for (auto think_info : think_infos_) {
        auto dfa = think_info.dfa_ptr;
        status.push_back(dfa ? dfa->status() : 0);
    }
    return status;
}

}  // namespace rtp_llm
