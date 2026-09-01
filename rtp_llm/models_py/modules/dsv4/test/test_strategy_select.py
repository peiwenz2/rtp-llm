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
)
from rtp_llm.models_py.modules.dsv4.moe.strategies import base as strategy_base
from rtp_llm.models_py.modules.dsv4.moe.strategies import select_strategy
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
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=True)
        self._env.start()
        strategy_base._DEPRECATION_WARNED.clear()
        strategy_base._MEGA_SE_FALLBACK_WARNED = False

    def tearDown(self):
        self._env.stop()

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
        # The topology invariant is checked independently of runtime probing.
        with mock.patch.object(MegaMoESEStrategy, "can_handle", return_value=True):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(
                    _cfg(ep_size=4, n_shared_experts=0), forced="mega_moe_se"
                )
        self.assertIn("requires exactly one shared expert", str(cm.exception))

    def test_auto_falls_back_to_mega_moe_when_se_is_unavailable(self):
        with mock.patch.object(
            MegaMoESEStrategy, "can_handle", return_value=False
        ), mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            self.assertIs(
                select_strategy(_cfg(ep_size=4, n_shared_experts=1)),
                MegaMoEStrategy,
            )

    def test_explicit_mega_moe_se_stays_strict_when_unavailable(self):
        with mock.patch.object(MegaMoESEStrategy, "can_handle", return_value=False):
            with self.assertRaisesRegex(
                RuntimeError, "Forced MoE strategy 'mega_moe_se'"
            ):
                select_strategy(
                    _cfg(ep_size=4, n_shared_experts=1), forced="mega_moe_se"
                )

    def test_auto_fails_after_both_mega_variants_are_unavailable(self):
        with mock.patch.object(
            MegaMoESEStrategy, "can_handle", return_value=False
        ), mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4, n_shared_experts=1))
        self.assertIn("mega_moe_se", str(cm.exception))
        self.assertIn("mega_moe", str(cm.exception))

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

    def test_legacy_strategy_names_map_to_public_names(self):
        for old, new in (
            ("mega", "mega_moe"),
            ("mega_se", "mega_moe_se"),
        ):
            with self.subTest(old=old), mock.patch.dict(
                os.environ, {"DSV4_MOE_STRATEGY": old}, clear=True
            ):
                strategy_base._DEPRECATION_WARNED.clear()
                with mock.patch.object(strategy_base.logging, "warning") as warning:
                    self.assertEqual(_resolve_forced(None), (new, True))
                warning.assert_called()

    def test_removed_legacy_fused_name_has_actionable_migration_error(self):
        with mock.patch.dict(
            os.environ, {"DSV4_MOE_STRATEGY": "mega_fused"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "MOE_STRATEGY=mega_moe_se"):
                _resolve_forced(None)

    def test_removed_constructor_fused_name_has_actionable_migration_error(self):
        with self.assertRaisesRegex(RuntimeError, "MOE_STRATEGY=mega_moe_se"):
            _resolve_forced("mega_fused")

    def test_legacy_boolean_toggles_keep_compatibility(self):
        cases = (
            ({"DSV4_USE_MEGA_MOE": "1"}, ("mega_moe", False)),
            ({"DSV4_USE_MEGA_MOE_SE": "1"}, ("mega_moe_se", True)),
            ({"DSV4_USE_GROUPED_FP4": "1"}, ("grouped_fp4", False)),
        )
        for env, expected in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                strategy_base._DEPRECATION_WARNED.clear()
                with mock.patch.object(strategy_base.logging, "warning") as warning:
                    self.assertEqual(_resolve_forced(None), expected)
                warning.assert_called()

    def test_legacy_negative_toggles_preserve_rollback_semantics(self):
        cases = (
            ("DSV4_USE_GROUPED_FP4", 1, ("local_loop", True)),
            ("DSV4_USE_MEGA_MOE_SE", 4, ("mega_moe", True)),
            ("DSV4_USE_MEGA_MOE_FUSED", 4, ("mega_moe", True)),
        )
        for name, ep_size, expected in cases:
            with self.subTest(name=name), mock.patch.dict(
                os.environ, {name: "0"}, clear=True
            ):
                strategy_base._DEPRECATION_WARNED.clear()
                with mock.patch.object(strategy_base.logging, "warning") as warning:
                    self.assertEqual(_resolve_forced(None, ep_size=ep_size), expected)
                warning.assert_called()

    def test_constructor_alias_warning_names_the_actual_source(self):
        with mock.patch.object(strategy_base.logging, "warning") as warning:
            self.assertEqual(_resolve_forced("mega"), ("mega_moe", True))
        self.assertIn("MoE strategy argument", warning.call_args.args[2])

    def test_legacy_and_public_conflict_fails(self):
        with mock.patch.dict(os.environ, {"DSV4_USE_GROUPED_FP4": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Conflicting MoE"):
                _resolve_forced("local_loop")

    def test_conflicting_legacy_toggles_fail(self):
        with mock.patch.dict(
            os.environ,
            {"DSV4_USE_MEGA_MOE_SE": "1", "DSV4_USE_GROUPED_FP4": "1"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "Conflicting legacy"):
                _resolve_forced(None)

    def test_invalid_legacy_toggle_fails(self):
        with mock.patch.dict(
            os.environ, {"DSV4_USE_MEGA_MOE": "sometimes"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "Invalid legacy MoE toggle"):
                _resolve_forced(None)

    def test_legacy_grouped_auto_preserves_auto_selection(self):
        with mock.patch.dict(
            os.environ, {"DSV4_USE_GROUPED_FP4": "auto"}, clear=True
        ), mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=True):
            forced, strict = _resolve_forced(None, ep_size=1)
            self.assertEqual((forced, strict), (None, False))
            self.assertIs(
                select_strategy(_cfg(ep_size=1), forced=forced, strict=strict),
                GroupedFP4Strategy,
            )

    def test_legacy_mega_toggles_are_noops_on_single_card(self):
        for name, value in (
            ("DSV4_USE_MEGA_MOE_SE", "0"),
            ("DSV4_USE_MEGA_MOE_FUSED", "0"),
        ):
            with self.subTest(name=name), mock.patch.dict(
                os.environ, {name: value}, clear=True
            ), mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=True):
                forced, strict = _resolve_forced(None, ep_size=1)
                self.assertEqual((forced, strict), (None, False))
                self.assertIs(
                    select_strategy(_cfg(ep_size=1), forced=forced, strict=strict),
                    GroupedFP4Strategy,
                )

    def test_legacy_grouped_disable_is_noop_on_ep_topology(self):
        with mock.patch.dict(
            os.environ, {"DSV4_USE_GROUPED_FP4": "0"}, clear=True
        ), mock.patch.object(MegaMoESEStrategy, "can_handle", return_value=True):
            forced, strict = _resolve_forced(None, ep_size=4)
            self.assertEqual((forced, strict), (None, False))
            self.assertIs(
                select_strategy(
                    _cfg(ep_size=4, n_shared_experts=1),
                    forced=forced,
                    strict=strict,
                ),
                MegaMoESEStrategy,
            )

    def test_legacy_se_enable_stays_strict_on_single_card(self):
        with mock.patch.dict(
            os.environ, {"DSV4_USE_MEGA_MOE_SE": "1"}, clear=True
        ), mock.patch.object(MegaMoESEStrategy, "can_handle", return_value=False):
            forced, strict = _resolve_forced(None, ep_size=1)
            self.assertEqual((forced, strict), ("mega_moe_se", True))
            with self.assertRaisesRegex(RuntimeError, "cannot handle cfg"):
                select_strategy(_cfg(ep_size=1), forced=forced, strict=strict)

    def test_removed_legacy_fused_toggle_has_actionable_migration_error(self):
        for ep_size in (1, 4):
            with self.subTest(ep_size=ep_size), mock.patch.dict(
                os.environ, {"DSV4_USE_MEGA_MOE_FUSED": "1"}, clear=True
            ):
                with self.assertRaisesRegex(RuntimeError, "MOE_STRATEGY=mega_moe_se"):
                    _resolve_forced(None, ep_size=ep_size)

    def test_legacy_fused_disable_preserves_non_fused_rollback(self):
        with mock.patch.dict(
            os.environ, {"DSV4_USE_MEGA_MOE_FUSED": "0"}, clear=True
        ), mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            forced, strict = _resolve_forced(None, ep_size=4)
            self.assertEqual((forced, strict), ("mega_moe", True))
            self.assertIs(
                select_strategy(
                    _cfg(ep_size=4, n_shared_experts=1),
                    forced=forced,
                    strict=strict,
                ),
                MegaMoEStrategy,
            )

    def test_legacy_grouped_disable_forces_local_loop(self):
        with mock.patch.dict(
            os.environ, {"DSV4_USE_GROUPED_FP4": "0"}, clear=True
        ), mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=True):
            forced, strict = _resolve_forced(None, ep_size=1)
            self.assertEqual((forced, strict), ("local_loop", True))
            self.assertIs(
                select_strategy(_cfg(ep_size=1), forced=forced, strict=strict),
                LocalLoopStrategy,
            )

    def test_legacy_mega_disable_fails_fast_on_ep_topology(self):
        with mock.patch.dict(os.environ, {"DSV4_USE_MEGA_MOE": "0"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "disables Mega MoE"):
                _resolve_forced(None, ep_size=4)

    def test_legacy_se_disable_excludes_fused_se(self):
        with mock.patch.dict(
            os.environ, {"DSV4_USE_MEGA_MOE_SE": "0"}, clear=True
        ), mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            forced, strict = _resolve_forced(None, ep_size=4)
            self.assertEqual((forced, strict), ("mega_moe", True))
            self.assertIs(
                select_strategy(
                    _cfg(ep_size=4, n_shared_experts=1),
                    forced=forced,
                    strict=strict,
                ),
                MegaMoEStrategy,
            )

    def test_public_strategy_cannot_override_legacy_negative_constraint(self):
        with mock.patch.dict(os.environ, {"DSV4_USE_GROUPED_FP4": "0"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Conflicting MoE"):
                _resolve_forced("grouped_fp4", ep_size=1)


if __name__ == "__main__":
    unittest.main()
