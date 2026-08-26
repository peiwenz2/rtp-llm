import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.base.rocm.select_topk import SelectTopk


@unittest.skipUnless(torch.version.hip is not None, "ROCm required")
class SelectTopkTest(unittest.TestCase):
    def test_eager_reuse_and_graph_isolation(self):
        op = SelectTopk(SimpleNamespace(moe_k=2))
        logits = torch.randn(4, 8, dtype=torch.bfloat16, device="cuda")
        ids = torch.empty(4, 2, dtype=torch.int32, device="cuda")
        weights = torch.empty(4, 2, dtype=torch.float32, device="cuda")
        op(logits, ids, weights)
        scratch_ptr = op._token_expert_indicies.data_ptr()
        op(logits[:1], ids[:1], weights[:1])
        self.assertEqual(op._token_expert_indicies.data_ptr(), scratch_ptr)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(logits, ids, weights)
        self.assertEqual(op._token_expert_indicies.data_ptr(), scratch_ptr)
        op(logits.repeat(2, 1), ids.repeat(2, 1), weights.repeat(2, 1))
        self.assertNotEqual(op._token_expert_indicies.data_ptr(), scratch_ptr)
        guard = torch.full_like(ids, -1)
        self.assertEqual(guard.data_ptr(), scratch_ptr)

        logits.copy_(torch.arange(32, dtype=torch.bfloat16, device="cuda").view(4, 8))
        graph.replay()
        self.assertTrue(torch.all(guard == -1))
        expected_weights, expected_ids = torch.softmax(logits.float(), dim=-1).topk(2)
        expected_weights /= expected_weights.sum(-1, keepdim=True)
        torch.testing.assert_close(ids, expected_ids.to(torch.int32))
        torch.testing.assert_close(weights, expected_weights)


if __name__ == "__main__":
    unittest.main()
