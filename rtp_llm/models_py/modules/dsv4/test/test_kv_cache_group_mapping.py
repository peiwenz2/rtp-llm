import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import INDEXER_KV, INDEXER_STATE
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    build_block_tables_batched,
)


class TestKVCacheGroupMapping(unittest.TestCase):

    def test_mtp_uses_layer_cache_physical_group_ids(self):
        layer_caches = [
            SimpleNamespace(
                region_name=INDEXER_KV,
                group_id=1,
                physical_group_id=5,
            ),
            SimpleNamespace(
                region_name=INDEXER_STATE,
                group_id=2,
                physical_group_id=6,
            ),
        ]
        kv_cache = SimpleNamespace(
            group_region_names=[
                0,
                0,
                INDEXER_KV,
                INDEXER_STATE,
                0,
                INDEXER_KV,
                INDEXER_STATE,
            ],
            layer_region_to_group_id=[[0, -1, -1, 1, 2, -1, -1, -1]],
            get_layer_caches=lambda _layer_id: layer_caches,
        )
        by_group = [
            torch.full((2, 1), group_id, dtype=torch.int32)
            for group_id in range(7)
        ]
        # These are the exact main INDEXER_KV kernel-page ids observed in the
        # cores. Interpreting local group 2 as draft INDEXER_STATE would trap.
        by_group[2] = torch.tensor([[192], [224]], dtype=torch.int32)
        attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device_by_group=by_group
        )

        block_tables = build_block_tables_batched(kv_cache, attn_inputs)

        self.assertIsNotNone(block_tables)
        self.assertIs(block_tables[INDEXER_KV], by_group[5])
        self.assertIs(block_tables[INDEXER_STATE], by_group[6])

    def test_fp8_graph_snapshot_indexes_global_tables(self):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplConfigFP8,
            DSv4DecodeFmhaImplFP8,
        )

        paged_pool_specs = {
            INDEXER_KV: (4, 128, 1),
            INDEXER_STATE: (4, 128, 1),
        }
        config = DSv4DecodeFmhaImplConfigFP8(
            max_batch_size=2,
            q_len=1,
            window_size=8,
            head_dim=32,
            max_seq_len=64,
            compress_ratios=[4],
            index_topk=4,
            paged_pool_specs=paged_pool_specs,
            paged_table_group_ids={
                INDEXER_KV: 5,
                INDEXER_STATE: 6,
            },
        )
        impl = object.__new__(DSv4DecodeFmhaImplFP8)
        impl.config = config
        impl._paged_entries_per_block = {
            attn_type: spec[0] for attn_type, spec in paged_pool_specs.items()
        }
        by_group = [
            torch.full((2, 1), group_id, dtype=torch.int32)
            for group_id in range(7)
        ]
        attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device_by_group=by_group
        )

        block_tables = impl._extract_paged_block_tables(attn_inputs)

        self.assertIs(block_tables[INDEXER_KV], by_group[5])
        self.assertIs(block_tables[INDEXER_STATE], by_group[6])


if __name__ == "__main__":
    unittest.main()
