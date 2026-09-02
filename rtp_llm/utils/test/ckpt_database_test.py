import json
import logging
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

from rtp_llm.utils.database import (
    CkptDatabase,
    FastSafeTensorsCompatibilityError,
    _callable_accepts_keyword,
    _fastsafetensors_stacked_moe_keyword,
)


class _FakeCkptFile:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name


class _FakeSingleGroup:
    def rank(self) -> int:
        return 0


def _install_fake_fastsafetensors(auto_loader_cls=None) -> types.ModuleType:
    module = types.ModuleType("fastsafetensors")
    module._rtp_test_fake = True
    module.SingleGroup = _FakeSingleGroup
    if auto_loader_cls is not None:
        module.AutoLoader = auto_loader_cls
    sys.modules["fastsafetensors"] = module
    return module


class CkptDataBaseTest(unittest.TestCase):

    def __init__(self, methodName: str = "Run CkptDataBaseTest") -> None:
        super().__init__(methodName)

    @staticmethod
    def _testdata_path():
        return os.path.join(
            os.getcwd(), "rtp_llm/utils/test/testdata/ckpt_database_testdata/"
        )

    def test_collect_ckpt_file(self):
        path = os.path.join(self._testdata_path(), "bin_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(
            path + "/pytorch_model.bin", database.pretrain_file_list[0].file_name
        )
        self.assertEqual(12, len(database.pretrain_file_list[0].get_tensor_names()))

        path = os.path.join(self._testdata_path(), "pt_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(path + "/test.pt", database.pretrain_file_list[0].file_name)
        self.assertEqual(36, len(database.pretrain_file_list[0].get_tensor_names()))

        path = os.path.join(self._testdata_path(), "safetensor_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(
            path + "/test.safetensors", database.pretrain_file_list[0].file_name
        )
        self.assertEqual(28, len(database.pretrain_file_list[0].get_tensor_names()))

        path = os.path.join(self._testdata_path(), "bin_testdata")
        lora_path = os.path.join(self._testdata_path(), "lora_testdata")
        database = CkptDatabase(path)
        database.load_lora("test", lora_path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(
            path + "/pytorch_model.bin", database.pretrain_file_list[0].file_name
        )
        self.assertEqual(12, len(database.pretrain_file_list[0].get_tensor_names()))
        self.assertEqual(1, len(database.lora_ckpt.LoraFileList))
        self.assertEqual(8, list(database.lora_ckpt.LoraFileList)[0].rank)
        self.assertEqual(8, list(database.lora_ckpt.LoraFileList)[0].lora_alpha)
        self.assertEqual(0.0, list(database.lora_ckpt.LoraFileList)[0].lora_dropout)
        self.assertEqual(
            ["c_proj", "w2", "c_attn", "w1"],
            list(database.lora_ckpt.LoraFileList)[0].target_modules,
        )
        self.assertEqual(1, len(list(database.lora_ckpt.LoraFileList.values())[0]))
        self.assertEqual(
            12,
            len(
                list(database.lora_ckpt.LoraFileList.values())[0][0].get_tensor_names()
            ),
        )

    def test_mix_ckpt_file(self):
        path = os.path.join(self._testdata_path(), "mixture_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(
            path + "/test.safetensors", database.pretrain_file_list[0].file_name
        )
        self.assertEqual(28, len(database.pretrain_file_list[0].get_tensor_names()))


class FastsafetensorsAutoLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._module_patch = patch.dict(sys.modules, {}, clear=False)
        self._module_patch.start()
        self.addCleanup(self._module_patch.stop)
        self._env_patch = patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)
        self._config_env_names = (
            "FASTSAFETENSORS_CONFIG",
            "FASTSAFETENSORS_CONFIG_JSON",
            "FASTSAFETENSORS_NOGDS",
        )
        for name in self._config_env_names:
            os.environ.pop(name, None)

    def test_split_templates_are_forwarded_and_prewrapped_keys_pass_through(
        self,
    ) -> None:
        closed = []
        observed_split_templates = []

        class FakeAutoLoader:
            def __init__(
                self,
                pg,
                files,
                device,
                local_copyout_filter=None,
                stacked_moe_tensors=None,
            ) -> None:
                observed_split_templates.append(stacked_moe_tensors)

            def iterate_weights(self):
                for expert_id in range(3):
                    yield f"experts.{expert_id}.weight", f"expert-{expert_id}"
                yield "plain", "plain-tensor"

            def close(self) -> None:
                closed.append(True)

        _install_fake_fastsafetensors(FakeAutoLoader)

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        result = list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                stacked_key_config={"stacked": "experts.{expert_id}.weight"},
            )
        )

        self.assertEqual(
            [key for key, _ in result],
            [
                "experts.0.weight",
                "experts.1.weight",
                "experts.2.weight",
                "plain",
            ],
        )
        self.assertEqual(
            [tensor for _, tensor in result[:3]],
            ["expert-0", "expert-1", "expert-2"],
        )
        self.assertEqual(result[3], ("plain", "plain-tensor"))
        self.assertEqual(
            observed_split_templates,
            [{"stacked": "experts.{expert_id}.weight"}],
        )
        self.assertEqual(closed, [True])

    def test_rank_local_copyout_filter_is_forwarded_to_auto_loader(self) -> None:
        observed_filters = []

        class FakeAutoLoader:
            def __init__(
                self,
                pg,
                files,
                device,
                local_copyout_filter=None,
                dim0_split_templates=None,
            ) -> None:
                observed_filters.append(local_copyout_filter)

            def iterate_weights(self):
                return iter(())

            def close(self) -> None:
                pass

        _install_fake_fastsafetensors(FakeAutoLoader)

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        required_keys = {"needed"}
        predicate = required_keys.__contains__

        list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                local_copyout_filter=predicate,
            )
        )

        self.assertEqual(observed_filters, [predicate])

    def test_legacy_dim0_split_keyword_is_forwarded_when_modern_name_is_absent(
        self,
    ) -> None:
        observed = []

        class FakeAutoLoader:
            def __init__(self, pg, files, device, dim0_split_templates=None) -> None:
                observed.append(dim0_split_templates)

            def iterate_weights(self):
                return iter(())

            def close(self) -> None:
                pass

        _install_fake_fastsafetensors(FakeAutoLoader)
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                stacked_key_config={"stacked": "experts.{expert_id}.weight"},
            )
        )

        self.assertEqual(observed, [{"stacked": "experts.{expert_id}.weight"}])

    def test_full_stacked_mode_disables_prebroadcast_split(self) -> None:
        observed_kwargs = []
        source_tensor = torch.tensor([[1, 2], [3, 4]])

        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                observed_kwargs.append(kwargs)

            def iterate_weights(self):
                yield "stacked", source_tensor

            def close(self) -> None:
                pass

        _install_fake_fastsafetensors(FakeAutoLoader)

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        result = list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                stacked_key_config={"stacked": "experts.{expert_id}.weight"},
                stacked_moe_mode="full-stacked",
            )
        )

        self.assertNotIn("stacked_moe_tensors", observed_kwargs[0])
        self.assertNotIn("dim0_split_templates", observed_kwargs[0])
        self.assertEqual(
            [name for name, _tensor in result],
            [
                "experts.0.weight",
                "experts.1.weight",
            ],
        )
        torch.testing.assert_close(result[0][1], torch.tensor([1, 2]))
        torch.testing.assert_close(result[1][1], torch.tensor([3, 4]))
        self.assertNotEqual(
            result[0][1].untyped_storage().data_ptr(),
            source_tensor.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            result[1][1].untyped_storage().data_ptr(),
            source_tensor.untyped_storage().data_ptr(),
        )

    def test_full_stacked_mode_clones_only_rank_local_experts(self) -> None:
        source_tensor = torch.tensor([[1, 2], [3, 4]])
        observed_filter_results = {}

        class FakeAutoLoader:
            def __init__(
                self, pg, files, device, local_copyout_filter=None, **kwargs
            ) -> None:
                self.local_copyout_filter = local_copyout_filter

            def iterate_weights(self):
                for key, tensor in (
                    ("unrelated", torch.tensor([9, 9])),
                    ("stacked", source_tensor),
                ):
                    accepted = self.local_copyout_filter(key)
                    observed_filter_results[key] = accepted
                    if accepted:
                        yield key, tensor

            def close(self) -> None:
                pass

        _install_fake_fastsafetensors(FakeAutoLoader)
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        local_keys = {"experts.1.weight"}

        result = list(
            database.fastsafetensors_weights_iterator(
                "cuda",
                stacked_key_config={"stacked": "experts.{expert_id}.weight"},
                local_copyout_filter=local_keys.__contains__,
                stacked_moe_mode="full-stacked",
            )
        )

        self.assertEqual(
            observed_filter_results,
            {"unrelated": False, "stacked": True},
        )
        self.assertEqual([name for name, _tensor in result], ["experts.1.weight"])
        torch.testing.assert_close(result[0][1], torch.tensor([3, 4]))

    def test_close_compatibility_failure_is_classified(self) -> None:
        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                pass

            def iterate_weights(self):
                yield "direct", torch.tensor([1])

            def close(self) -> None:
                raise AttributeError("incompatible close API")

        _install_fake_fastsafetensors(FakeAutoLoader)
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertRaisesRegex(
            FastSafeTensorsCompatibilityError,
            "failed to close FastSafeTensors AutoLoader",
        ):
            list(database.fastsafetensors_weights_iterator("cuda"))

    def test_close_failure_does_not_replace_checkpoint_error(self) -> None:
        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                pass

            def iterate_weights(self):
                raise FileNotFoundError("checkpoint shard disappeared")
                yield

            def close(self) -> None:
                raise AttributeError("incompatible close API")

        _install_fake_fastsafetensors(FakeAutoLoader)
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertLogs(level="WARNING") as logs:
            with self.assertRaisesRegex(
                FileNotFoundError, "checkpoint shard disappeared"
            ):
                list(database.fastsafetensors_weights_iterator("cuda"))

        self.assertIn("close failed while preserving", "\n".join(logs.output))

    def test_wrapper_without_auto_loader_fails_instead_of_legacy_fallback(self) -> None:
        _install_fake_fastsafetensors()

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertRaisesRegex(FastSafeTensorsCompatibilityError, "AutoLoader"):
            list(database.fastsafetensors_weights_iterator("cuda"))

    def test_constructor_abi_error_is_classified_as_compatibility_failure(self):
        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                raise RuntimeError(
                    "Incompatible fast_safetensors native wheel; missing fuse-shm APIs"
                )

        _install_fake_fastsafetensors(FakeAutoLoader)
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertRaisesRegex(FastSafeTensorsCompatibilityError, "native wheel"):
            list(database.fastsafetensors_weights_iterator("cuda"))

    def test_constructor_checkpoint_io_error_remains_fail_fast(self):
        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                raise FileNotFoundError("model.safetensors disappeared")

        _install_fake_fastsafetensors(FakeAutoLoader)
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertRaisesRegex(FileNotFoundError, "disappeared"):
            list(database.fastsafetensors_weights_iterator("cuda"))

    def test_constructor_dlopen_oserror_is_a_compatibility_failure(self):
        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                raise OSError("cannot open shared object file: libfastsafetensors.so")

        _install_fake_fastsafetensors(FakeAutoLoader)
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertRaisesRegex(FastSafeTensorsCompatibilityError, "shared object"):
            list(database.fastsafetensors_weights_iterator("cuda"))

    def test_checkpoint_iteration_error_is_not_reclassified(self):
        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                pass

            def iterate_weights(self):
                raise RuntimeError("checkpoint tensor shape mismatch")

            def close(self) -> None:
                pass

        _install_fake_fastsafetensors(FakeAutoLoader)
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            list(database.fastsafetensors_weights_iterator("cuda"))

    def test_iteration_abi_error_is_classified_and_loader_is_closed(self):
        closed = []

        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                pass

            def iterate_weights(self):
                raise RuntimeError("undefined symbol: fast_safetensors_reader")

            def close(self) -> None:
                closed.append(True)

        _install_fake_fastsafetensors(FakeAutoLoader)
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertRaisesRegex(
            FastSafeTensorsCompatibilityError, "undefined symbol"
        ):
            list(database.fastsafetensors_weights_iterator("cuda"))
        self.assertEqual(closed, [True])

    def test_iteration_checkpoint_io_error_remains_fail_fast_and_closes_loader(self):
        closed = []

        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                pass

            def iterate_weights(self):
                raise OSError("I/O error while reading model.safetensors")

            def close(self) -> None:
                closed.append(True)

        _install_fake_fastsafetensors(FakeAutoLoader)
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        with self.assertRaisesRegex(OSError, "I/O error"):
            list(database.fastsafetensors_weights_iterator("cuda"))
        self.assertEqual(closed, [True])

    def test_per_expert_mode_uses_full_stacked_when_wrapper_lacks_split_capability(
        self,
    ) -> None:
        source_tensor = torch.tensor([[1, 2], [3, 4]])

        class FakeAutoLoader:
            def __init__(self, pg, files, device) -> None:
                pass

            def iterate_weights(self):
                yield "stacked", source_tensor

            def close(self) -> None:
                pass

        _install_fake_fastsafetensors(FakeAutoLoader)

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        local_keys = {"stacked", "experts.1.weight"}

        with self.assertLogs(level="WARNING") as logs:
            result = list(
                database.fastsafetensors_weights_iterator(
                    "cuda",
                    stacked_key_config={"stacked": "experts.{expert_id}.weight"},
                    local_copyout_filter=local_keys.__contains__,
                )
            )
        self.assertEqual(
            [name for name, _tensor in result],
            ["experts.1.weight"],
        )
        self.assertIn("full-stacked", "\n".join(logs.output))
        self.assertIn("local_copyout_filter", "\n".join(logs.output))
        self.assertIn("requested_mode=per-expert", "\n".join(logs.output))
        self.assertIn("effective_mode=full-stacked", "\n".join(logs.output))
        self.assertIn("effective_mode=consumer-filter", "\n".join(logs.output))
        self.assertIn("degraded_reason=", "\n".join(logs.output))

    def test_legacy_nogds_overrides_config_json(self) -> None:
        observed_config = []

        class FakeAutoLoader:
            def __init__(
                self,
                pg,
                files,
                device,
                local_copyout_filter=None,
                stacked_moe_tensors=None,
            ) -> None:
                observed_config.append(
                    json.loads(os.environ["FASTSAFETENSORS_CONFIG_JSON"])
                )

            def iterate_weights(self):
                return iter(())

            def close(self) -> None:
                pass

        _install_fake_fastsafetensors(FakeAutoLoader)

        os.environ["FASTSAFETENSORS_CONFIG_JSON"] = json.dumps({"loader": "fuse-shm"})
        os.environ["FASTSAFETENSORS_NOGDS"] = "1"

        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]
        list(database.fastsafetensors_weights_iterator("cuda"))

        self.assertEqual(
            json.loads(os.environ["FASTSAFETENSORS_CONFIG_JSON"]),
            {"loader": "base", "base": {"copier_type": "nogds"}},
        )
        self.assertEqual(
            observed_config,
            [{"loader": "base", "base": {"copier_type": "nogds"}}],
        )

    def test_without_legacy_nogds_preserves_inline_config(self) -> None:
        observed_config_json = []

        class FakeAutoLoader:
            def __init__(self, pg, files, device, **kwargs) -> None:
                observed_config_json.append(os.environ["FASTSAFETENSORS_CONFIG_JSON"])

            def iterate_weights(self):
                return iter(())

            def close(self) -> None:
                pass

        _install_fake_fastsafetensors(FakeAutoLoader)
        expected = json.dumps({"loader": "fuse-shm"})
        os.environ["FASTSAFETENSORS_CONFIG_JSON"] = expected
        os.environ["FASTSAFETENSORS_NOGDS"] = "0"
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_FakeCkptFile("model.safetensors")]

        list(database.fastsafetensors_weights_iterator("cuda"))

        self.assertEqual(os.environ["FASTSAFETENSORS_CONFIG_JSON"], expected)
        self.assertEqual(observed_config_json, [expected])

    def test_database_rejects_unknown_stacked_moe_mode(self) -> None:
        database = object.__new__(CkptDatabase)

        with self.assertRaisesRegex(ValueError, "per-expert.*full-stacked"):
            database.fastsafetensors_weights_iterator(
                "cuda", stacked_moe_mode="surprise"
            )


class InstalledFastsafetensorsContractTest(unittest.TestCase):
    def setUp(self) -> None:
        installed_module = sys.modules.get("fastsafetensors")
        self.assertFalse(getattr(installed_module, "_rtp_test_fake", False))

    def test_available_auto_loader_contract_is_classified(self) -> None:
        expected_tier = os.environ.get("RTP_LLM_EXPECT_FASTSAFETENSORS_TIER")
        try:
            import fastsafetensors
        except ImportError as error:
            actual_tier = "scratch"
            if expected_tier is not None:
                self.assertEqual(actual_tier, expected_tier, str(error))
                return
            self.skipTest(f"fastsafetensors is not installed: {error}")
        auto_loader = getattr(fastsafetensors, "AutoLoader", None)
        load_config = getattr(fastsafetensors, "load_config", None)
        if auto_loader is None:
            actual_tier = "scratch"
        else:
            if not _callable_accepts_keyword(
                auto_loader.__init__, "local_copyout_filter"
            ):
                actual_tier = "consumer-filter"
            elif _fastsafetensors_stacked_moe_keyword(auto_loader.__init__) is None:
                actual_tier = "full-stacked"
            else:
                actual_tier = "per-expert"

        logging.info("RTP FastSafeTensors capability tier: %s", actual_tier)
        if expected_tier is not None:
            self.assertEqual(actual_tier, expected_tier)
            if actual_tier != "per-expert":
                return
        elif actual_tier != "per-expert":
            self.skipTest(f"installed wheel uses the {actual_tier} compatibility path")

        if load_config is None:
            self.skipTest(
                "installed wheel supports per-expert delivery with legacy memory budget"
            )
        config = load_config()
        self.assertTrue(hasattr(config, "estimated_peak_device_bytes"))
        estimate = config.estimated_peak_device_bytes
        if estimate is not None:
            self.assertIsInstance(estimate, (int, float))
            self.assertGreater(estimate, 0)


class LoraTest(unittest.TestCase):

    def __init__(self, methodName: str = "Run CkptDataBaseTest") -> None:
        super().__init__(methodName)

    @staticmethod
    def _testdata_path():
        return os.path.join(
            os.getcwd(), "rtp_llm/utils/test/testdata/ckpt_database_testdata/"
        )

    def test_collect_ckpt_file(self):
        path = os.path.join(self._testdata_path(), "bin_testdata")
        database = CkptDatabase(path)
        self.assertEqual(1, len(database.pretrain_file_list))
        self.assertEqual(
            path + "/pytorch_model.bin", database.pretrain_file_list[0].file_name
        )
        self.assertEqual(12, len(database.pretrain_file_list[0].get_tensor_names()))

        lora_path = os.path.join(self._testdata_path(), "lora_testdata")
        database.load_lora("test_name", lora_path)
        self.assertEqual(1, len(database.lora_ckpt.LoraFileList))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(8, lora_config.rank)
        self.assertEqual(8, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual(["c_proj", "w2", "c_attn", "w1"], lora_config.target_modules)
        self.assertEqual(1, len(database.lora_ckpt.get_lora("test_name")))
        self.assertEqual(12, len(database.get_lora_tensor_names("test_name")))

        self.assertTrue(database.remove_lora("test_name"))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(0, lora_config.rank)
        self.assertEqual(0, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual([], lora_config.target_modules)
        self.assertEqual(0, len(database.lora_ckpt.get_lora("test_name")))
        self.assertEqual(0, len(database.get_lora_tensor_names("test_name")))

        lora_path = os.path.join(self._testdata_path(), "lora_testdata_safetensor")
        database.load_lora("test_name", lora_path)
        self.assertEqual(1, len(database.lora_ckpt.LoraFileList))
        lora_config = database.get_lora_config("test_name")
        self.assertEqual(8, lora_config.rank)
        self.assertEqual(8, lora_config.lora_alpha)
        self.assertEqual(0.0, lora_config.lora_dropout)
        self.assertEqual(["c_proj", "w2", "c_attn", "w1"], lora_config.target_modules)
        self.assertEqual(1, len(database.lora_ckpt.get_lora("test_name")))
        self.assertEqual(12, len(database.get_lora_tensor_names("test_name")))


class TensorIndexTest(unittest.TestCase):
    """Tests for the O(1) _tensor_index lookup introduced in CkptDatabase."""

    @staticmethod
    def _testdata_path():
        return os.path.join(
            os.getcwd(), "rtp_llm/utils/test/testdata/ckpt_database_testdata/"
        )

    def test_tensor_index_lookup(self):
        path = os.path.join(self._testdata_path(), "safetensor_testdata")
        database = CkptDatabase(path)

        # _tensor_index should contain all tensor names
        all_names = database.get_pretrain_tensor_names()
        for name in all_names:
            self.assertIn(name, database._tensor_index)

        # has_tensor should return True for known tensors, False for unknown
        self.assertTrue(database.has_tensor(all_names[0]))
        self.assertFalse(database.has_tensor("nonexistent.weight"))

        # load_tensor should return a non-empty list for known tensors
        result = database.load_tensor(all_names[0])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], torch.Tensor)

        # load_tensor for unknown tensors should return empty list
        result = database.load_tensor("nonexistent.weight")
        self.assertEqual(len(result), 0)

    def test_tensor_index_cleanup(self):
        path = os.path.join(self._testdata_path(), "safetensor_testdata")
        database = CkptDatabase(path)

        self.assertGreater(len(database._tensor_index), 0)
        database._tensor_index.clear()
        self.assertEqual(len(database._tensor_index), 0)
        # After clearing, has_tensor should return False
        all_names = database.get_pretrain_tensor_names()
        self.assertFalse(database.has_tensor(all_names[0]))


class SafetensorHandleCacheTest(unittest.TestCase):
    """Tests for CkptFileInfo safetensor handle caching."""

    @staticmethod
    def _testdata_path():
        return os.path.join(
            os.getcwd(), "rtp_llm/utils/test/testdata/ckpt_database_testdata/"
        )

    def test_handle_cache_returns_same_object(self):
        from rtp_llm.utils.ckpt_file_info import CkptFileInfo

        path = os.path.join(
            self._testdata_path(), "safetensor_testdata", "test.safetensors"
        )
        info = CkptFileInfo(file_name=path)

        h1 = info._get_safetensor_handle()
        h2 = info._get_safetensor_handle()
        self.assertIs(h1, h2)

    def test_close_handle_clears_cache(self):
        from rtp_llm.utils.ckpt_file_info import CkptFileInfo

        path = os.path.join(
            self._testdata_path(), "safetensor_testdata", "test.safetensors"
        )
        info = CkptFileInfo(file_name=path)

        info._get_safetensor_handle()
        self.assertIsNotNone(info._st_handle)

        info.close_safetensor_handle()
        self.assertIsNone(info._st_handle)

        # Can reopen after close
        h = info._get_safetensor_handle()
        self.assertIsNotNone(h)
        info.close_safetensor_handle()


import torch

if __name__ == "__main__":
    unittest.main()
