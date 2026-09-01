"""Host tests for model-aware DeepSeek-V4 MoE strategy selection."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    DeepEPStrategy,
    GroupedFP4Strategy,
    LocalLoopStrategy,
    MegaMoESEStrategy,
    MegaMoEStrategy,
    MoeCfg,
    _has_fp8_fp4_grouped_kernel,
    select_strategy,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import _resolve_forced


def _cfg(ep_size: int = 1, n_shared_experts: int = 1) -> MoeCfg:
    n_local = 256 // max(ep_size, 1)
    return MoeCfg(
        layer_id=2,
        dim=7168,
        moe_inter_dim=2048,
        n_routed_experts=256,
        n_activated_experts=6,
        n_shared_experts=n_shared_experts,
        swiglu_limit=10.0,
        ep_size=ep_size,
        ep_rank=0,
        n_local_experts=n_local,
        local_expert_start=0,
        local_expert_end=n_local,
        max_tokens_per_rank=8192,
    )


class StrategySelectTest(unittest.TestCase):
    def test_ep1_with_grouped_kernel_picks_grouped(self):
        with mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=True
        ), mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            self.assertIs(select_strategy(_cfg(ep_size=1)), GroupedFP4Strategy)

    def test_grouped_selection_is_gated_by_ep_size(self):
        cfg = _cfg(ep_size=2)
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "_has_fp8_fp4_grouped_kernel",
            return_value=True,
        ):
            self.assertFalse(GroupedFP4Strategy.can_handle(cfg))

    def test_grouped_kernel_probe_requires_sm100_family(self):
        fake_deep_gemm = types.SimpleNamespace(
            m_grouped_fp8_fp4_gemm_nt_contiguous=object(),
            get_mk_alignment_for_contiguous_layout=lambda: (128, 128),
        )
        with mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.get_device_capability",
            return_value=(12, 0),
        ):
            self.assertFalse(_has_fp8_fp4_grouped_kernel())

        with mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.get_device_capability",
            return_value=(10, 3),
        ):
            self.assertTrue(_has_fp8_fp4_grouped_kernel())

    def test_ep1_no_grouped_falls_to_local(self):
        with mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=False
        ), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=False
        ), mock.patch.object(
            MegaMoESEStrategy, "can_handle", return_value=False
        ), mock.patch.object(
            DeepEPStrategy, "can_handle", return_value=False
        ):
            self.assertIs(select_strategy(_cfg(ep_size=1)), LocalLoopStrategy)

    def test_ep_gt1_with_shared_expert_defaults_to_mega_moe_se(self):
        with mock.patch.object(MegaMoESEStrategy, "can_handle", return_value=True):
            self.assertIs(
                select_strategy(_cfg(ep_size=4, n_shared_experts=1)),
                MegaMoESEStrategy,
            )

    def test_ep_gt1_without_shared_expert_defaults_to_mega_moe(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            self.assertIs(
                select_strategy(_cfg(ep_size=4, n_shared_experts=0)),
                MegaMoEStrategy,
            )

    def test_explicit_mega_moe_wins_even_when_model_has_shared_expert(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            self.assertIs(
                select_strategy(_cfg(ep_size=4, n_shared_experts=1), forced="mega_moe"),
                MegaMoEStrategy,
            )

    def test_explicit_mega_moe_se_selects_fused_shared_expert(self):
        with mock.patch.object(MegaMoESEStrategy, "can_handle", return_value=True):
            self.assertIs(
                select_strategy(
                    _cfg(ep_size=4, n_shared_experts=1), forced="mega_moe_se"
                ),
                MegaMoESEStrategy,
            )

    def test_explicit_mega_moe_se_without_shared_expert_fails(self):
        with mock.patch.object(MegaMoESEStrategy, "can_handle", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(
                    _cfg(ep_size=4, n_shared_experts=0), forced="mega_moe_se"
                )
        self.assertIn("Forced MoE strategy 'mega_moe_se'", str(cm.exception))

    def test_removed_selection_envs_do_not_override_explicit_or_auto(self):
        legacy_env = {
            "DSV4_MOE_STRATEGY": "mega_moe_se",
            "DSV4_USE_MEGA_MOE": "0",
            "DSV4_USE_MEGA_MOE_SE": "0",
            "DSV4_USE_GROUPED_FP4": "1",
        }
        with mock.patch.dict(os.environ, legacy_env, clear=False), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=True
        ), mock.patch.object(MegaMoESEStrategy, "can_handle", return_value=True):
            self.assertIs(
                select_strategy(_cfg(ep_size=4, n_shared_experts=1), forced="mega_moe"),
                MegaMoEStrategy,
            )
            self.assertIs(
                select_strategy(_cfg(ep_size=4, n_shared_experts=1)),
                MegaMoESEStrategy,
            )

    def test_model_selected_mega_moe_se_unavailable_fails_without_fallback(self):
        with mock.patch.object(MegaMoESEStrategy, "can_handle", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4, n_shared_experts=1))
        self.assertIn("selected 'mega_moe_se' from model metadata", str(cm.exception))
        self.assertIn("fallback is disabled", str(cm.exception))

    def test_model_selected_mega_moe_unavailable_fails_without_fallback(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4, n_shared_experts=0))
        self.assertIn("selected 'mega_moe' from model metadata", str(cm.exception))
        self.assertIn("fallback is disabled", str(cm.exception))

    def test_forced_known_and_capable_returns_it(self):
        self.assertIs(
            select_strategy(_cfg(ep_size=1), forced="local_loop"),
            LocalLoopStrategy,
        )

    def test_forced_known_but_incapable_raises(self):
        with mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=1), forced="grouped_fp4")
        self.assertIn("Forced MoE strategy 'grouped_fp4'", str(cm.exception))

    def test_forced_ep_gt1_non_mega_raises_even_if_capable(self):
        with mock.patch.object(DeepEPStrategy, "can_handle", return_value=True):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4), forced="deepep")
        self.assertIn("requires MegaMoEStrategy", str(cm.exception))

    def test_forced_unknown_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            select_strategy(_cfg(), forced="bogus")
        self.assertIn("Unknown MoE strategy 'bogus'", str(cm.exception))

    def test_resolve_auto_values(self):
        self.assertEqual(_resolve_forced(None), (None, False))
        self.assertEqual(_resolve_forced(""), (None, False))
        self.assertEqual(_resolve_forced("auto"), (None, False))

    def test_resolve_named_value_is_strict(self):
        self.assertEqual(_resolve_forced("mega_moe"), ("mega_moe", True))
        self.assertEqual(_resolve_forced("mega_moe_se"), ("mega_moe_se", True))


if __name__ == "__main__":
    unittest.main()
