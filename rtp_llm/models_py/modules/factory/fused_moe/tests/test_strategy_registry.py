"""Strategy-registry diagnostics for public MOE_STRATEGY values."""

import unittest
from types import SimpleNamespace

from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)


def _config(strategy: str):
    return SimpleNamespace(
        model_config=SimpleNamespace(quant_config=None, model_type="test_model"),
        moe_config=SimpleNamespace(use_deepep_low_latency=False),
        moe_strategy=strategy,
        ep_size=1,
        world_size=1,
        tp_size=1,
    )


class StrategyRegistryDiagnosticsTest(unittest.TestCase):
    def test_request_names_value_and_current_model_scope(self):
        with self.assertRaises(ValueError) as cm:
            StrategyRegistry().get_strategy(_config("grouped_fp4"))

        message = str(cm.exception)
        self.assertIn("MOE_STRATEGY='grouped_fp4'", message)
        self.assertIn("model scope 'test_model'", message)
        self.assertIn("generic fused-MoE factory", message)


if __name__ == "__main__":
    unittest.main()
