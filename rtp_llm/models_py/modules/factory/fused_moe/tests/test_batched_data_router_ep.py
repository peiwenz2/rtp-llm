import unittest
from typing import Optional
from unittest import mock

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.factory.fused_moe.defs import config_adapter
from rtp_llm.models_py.modules.factory.fused_moe.defs import fused_moe as moe_defs
from rtp_llm.models_py.modules.factory.fused_moe.defs import quant_config
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.executor.batched_triton_executor import (
    BatchedTritonExperts,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.router import (
    batched_data_router,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

EXPERT_NUM = 8
TP_SIZE = 2
EP_RANK = 1
LOCAL_EXPERTS = EXPERT_NUM // TP_SIZE
LOCAL_LO = LOCAL_EXPERTS * EP_RANK
TOP_K = 2
HIDDEN = 4
NUM_TOKENS = 6


class BatchedDataRouterEpTest(unittest.TestCase):
    """Covers the non-local-expert branch that tp_size == ep_size == 1 hides:
    with ep_rank > 0 most top-k slots fall outside this rank, so ``routed`` is a
    real mask over the scratch column and over finalize's zeroing."""

    @staticmethod
    def _make_config(max_tokens: int) -> config_adapter.MoEConfigAdapter:
        model_config = ModelConfig()
        model_config.hidden_size = HIDDEN
        model_config.expert_num = EXPERT_NUM
        model_config.moe_k = TOP_K
        parallelism = ParallelismConfig()
        parallelism.tp_size = TP_SIZE
        parallelism.ep_size = TP_SIZE
        parallelism.ep_rank = EP_RANK
        moe_config = MoeConfig()
        moe_config.ll_num_max_token = max_tokens
        return config_adapter.MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism,
            moe_config=moe_config,
        )

    @classmethod
    def _make_router(cls, max_tokens: int) -> batched_data_router.BatchedDataRouter:
        return batched_data_router.BatchedDataRouter(
            config=cls._make_config(max_tokens),
            quant_config=quant_config.FusedMoEQuantConfig(quant_dtype=None),
        )

    def setUp(self) -> None:
        self.router = self._make_router(NUM_TOKENS)
        torch.manual_seed(0)
        self.a1 = torch.arange(NUM_TOKENS * HIDDEN, dtype=torch.float32).view(
            NUM_TOKENS, HIDDEN
        )
        # Boundary ids: LOCAL_LO-1 is the last non-local id, LOCAL_LO the first
        # local one. Local expert 0 stays empty so its placeholder row is poison.
        self.topk_ids = torch.tensor(
            [(LOCAL_LO - 1, LOCAL_LO + 1), (LOCAL_LO + 1, EXPERT_NUM - 1)]
            + [(0, LOCAL_LO + 2), (0, LOCAL_LO - 1)]
            + [(EXPERT_NUM - 1, LOCAL_LO + 1), (LOCAL_LO + 2, 1)],
            dtype=torch.int32,
        )
        self.topk_weights = torch.rand(NUM_TOKENS, TOP_K) + 0.5
        self.local = (self.topk_ids >= LOCAL_LO) & (self.topk_ids < EXPERT_NUM)
        self.payload = self._prepare(self.a1, self.topk_weights, self.topk_ids)
        meta = self.payload.expert_tokens_meta
        assert meta is not None and meta.expert_num_tokens is not None
        self.counts = meta.expert_num_tokens

    def _prepare(
        self, a1: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
    ) -> moe_defs.ExpertForwardPayload:
        return self.router.prepare(a1, None, None, weights, ids)

    def _finalize(
        self, expert_output: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        reducer = mock.MagicMock(side_effect=lambda t, _g: t)
        with mock.patch.object(batched_data_router, "all_reduce", reducer):
            result = self.router.finalize(
                moe_defs.CombineForwardPayload(fused_expert_output=expert_output),
                self.topk_weights if weights is None else weights,
                self.topk_ids,
                False,
                None,
            )
        reducer.assert_called_once()
        self.assertIs(reducer.call_args.args[1], Group.TP)
        return result

    def test_plan_packs_every_local_slot_exactly_once(self) -> None:
        for e in range(LOCAL_EXPERTS):
            tokens = (self.topk_ids == e + LOCAL_LO).any(dim=1).nonzero().flatten()
            self.assertEqual(int(self.counts[e]), tokens.numel(), f"expert {e} count")
            # Exact, not a multiset compare: packed row ranks fix token order
            # inside each expert block and _packed_rows depends on it.
            packed = self.payload.expert_x[e, : tokens.numel()]
            self.assertTrue(torch.equal(packed, self.a1[tokens]), f"expert {e} rows")

    def test_finalize_ignores_poisoned_padding(self) -> None:
        """Rows past each expert's count are never gathered, so a non-finite tail
        cannot reach the output. Token 3 has no local slot at all, so it must come
        out exactly zero even though the placeholder row it points at holds NaN --
        folding the mask into the weights instead would leave NaN * 0 == NaN."""
        expert_output = torch.zeros(LOCAL_EXPERTS, NUM_TOKENS, HIDDEN)
        for e in range(LOCAL_EXPERTS):
            valid = int(self.counts[e])
            expert_output[e, :valid] = self.payload.expert_x[e, :valid]
            expert_output[e, valid:] = float("inf") if e % 2 else float("nan")

        out = self._finalize(expert_output)

        self.assertTrue(torch.isfinite(out).all(), "padding leaked into the output")
        expected = self.a1 * (self.local * self.topk_weights).sum(1, keepdim=True)
        torch.testing.assert_close(out, expected)

    def test_finalize_requires_a_fresh_plan(self) -> None:
        """A consumed plan, and one a failed prepare dropped, both have to be
        rejected rather than combined against stale rows."""
        expert_output = torch.zeros(LOCAL_EXPERTS, NUM_TOKENS, HIDDEN)
        self._finalize(expert_output)
        with self.assertRaisesRegex(RuntimeError, r"finalize\(\) called before"):
            self._finalize(expert_output)

        big = torch.zeros((NUM_TOKENS + 1, TOP_K), dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, r"supports at most"):
            self._prepare(torch.zeros((NUM_TOKENS + 1, HIDDEN)), big.float(), big)
        with self.assertRaisesRegex(RuntimeError, r"finalize\(\) called before"):
            self._finalize(expert_output)

    def test_prepare_rejects_reentrant_use(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"prepare\(\) called before"):
            self._prepare(self.a1, self.topk_weights, self.topk_ids)
        self._finalize(torch.zeros(LOCAL_EXPERTS, NUM_TOKENS, HIDDEN))

    def test_executor_failure_does_not_poison_router(self) -> None:
        router = self._make_router(NUM_TOKENS)
        experts = BatchedTritonExperts(
            self._make_config(NUM_TOKENS),
            quant_config.FusedMoEQuantConfig(quant_dtype=None),
            {
                W.moe_w1: torch.empty(LOCAL_EXPERTS, 8, HIDDEN),
                W.moe_w2: torch.empty(LOCAL_EXPERTS, HIDDEN, 4),
            },
        )
        fused_moe = moe_defs.FusedMoe(router, experts, EXPERT_NUM)

        with self.assertRaisesRegex(NotImplementedError, "expert_map"):
            fused_moe(
                self.a1,
                self.topk_weights,
                self.topk_ids,
                expert_map=torch.arange(EXPERT_NUM),
            )

        payload = router.prepare(self.a1, None, None, self.topk_weights, self.topk_ids)
        with mock.patch.object(
            batched_data_router, "all_reduce", side_effect=lambda t, _g: t
        ):
            out = router.finalize(
                moe_defs.CombineForwardPayload(
                    fused_expert_output=payload.expert_x.clone()
                ),
                self.topk_weights,
                self.topk_ids,
                False,
                None,
            )

        expected = self.a1 * (self.local * self.topk_weights).sum(1, keepdim=True)
        torch.testing.assert_close(out, expected)

    def test_rejected_reentrant_forward_does_not_discard_the_active_plan(self) -> None:
        experts = mock.Mock(spec=moe_defs.FusedMoeExpertExecutor)
        fused_moe = moe_defs.FusedMoe(self.router, experts, EXPERT_NUM)

        with self.assertRaisesRegex(RuntimeError, r"prepare\(\) called before"):
            fused_moe(self.a1, self.topk_weights, self.topk_ids)

        out = self._finalize(self.payload.expert_x.clone())
        experts.execute.assert_not_called()
        expected = self.a1 * (self.local * self.topk_weights).sum(1, keepdim=True)
        torch.testing.assert_close(out, expected)

    def test_round_trip_matches_reference_across_token_counts(self) -> None:
        """The fixed 6-token fixture cannot catch an off-by-one that only shows
        up at another count, and the earlier sweep covered the since-replaced
        argsort pack. Duplicate experts inside one token's top-k occur naturally
        at these sizes and are covered by the same reference."""
        for n in (1, 4, 14, 29, 32, 33, 37):
            with self.subTest(num_tokens=n):
                router = self._make_router(n)
                a1 = torch.randn(n, HIDDEN)
                ids = torch.randint(0, EXPERT_NUM, (n, TOP_K), dtype=torch.int32)
                weights = torch.rand(n, TOP_K) + 0.5

                payload = router.prepare(a1, None, None, weights, ids)
                meta = payload.expert_tokens_meta
                assert meta is not None and meta.expert_num_tokens is not None

                expert_output = torch.zeros(LOCAL_EXPERTS, n, HIDDEN)
                for e in range(LOCAL_EXPERTS):
                    tokens = (ids == e + LOCAL_LO).any(dim=1).nonzero().flatten()
                    self.assertEqual(
                        int(meta.expert_num_tokens[e]), tokens.numel(), f"expert {e}"
                    )
                    live = tokens.numel()
                    expert_output[e, :live] = payload.expert_x[e, :live]
                    expert_output[e, live:] = float("nan")

                reducer = mock.MagicMock(side_effect=lambda t, _g: t)
                with mock.patch.object(batched_data_router, "all_reduce", reducer):
                    out = router.finalize(
                        moe_defs.CombineForwardPayload(
                            fused_expert_output=expert_output
                        ),
                        weights,
                        ids,
                        False,
                        None,
                    )

                local = (ids >= LOCAL_LO) & (ids < EXPERT_NUM)
                expected = a1 * (local * weights).sum(1, keepdim=True)
                torch.testing.assert_close(out, expected)

    def test_zero_tokens_round_trips(self) -> None:
        """Empty batch is reachable on a DP rank; deriving the counts from the
        last cumsum row used to raise IndexError."""
        # Release setUp's plan first: prepare() is non-reentrant.
        self._finalize(torch.zeros(LOCAL_EXPERTS, NUM_TOKENS, HIDDEN))
        empty = torch.zeros((0, TOP_K))
        payload = self._prepare(torch.zeros((0, HIDDEN)), empty, empty.to(torch.int32))
        meta = payload.expert_tokens_meta
        assert meta is not None and meta.expert_num_tokens is not None

        self.assertEqual(payload.expert_x.shape, (LOCAL_EXPERTS, 0, HIDDEN))
        self.assertTrue(
            torch.equal(
                meta.expert_num_tokens, torch.zeros(LOCAL_EXPERTS, dtype=torch.int32)
            )
        )
        out = self._finalize(torch.zeros((LOCAL_EXPERTS, 0, HIDDEN)), empty)
        self.assertEqual(out.shape, (0, HIDDEN))


if __name__ == "__main__":
    unittest.main()
