"""Every case here is a way a setting reached the engine and did nothing.

Needs tensorrt_llm, so it runs inside the container like the rest of the
scripts/tests suite.
"""

import os
import sys
import unittest

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validate_engine_yaml import check


def problems(text: str) -> list[str]:
    return check(yaml.safe_load(text) or {})


class EngineYamlValidationTests(unittest.TestCase):
    def test_the_configuration_this_repo_ships_is_accepted(self):
        self.assertEqual(problems("""
attn_backend: TRTLLM
moe_config:
  backend: AUTO
cuda_graph_config:
  mode: decode
  max_batch_size: 64
  enable_padding: true
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 4096
"""), [])

    def test_a_misspelt_key_is_rejected_with_a_suggestion(self):
        found = problems("cuda_graph_confgi:\n  max_batch_size: 64")
        self.assertTrue(found)
        self.assertIn("cuda_graph_config", found[0])

    def test_cuda_graph_config_without_its_discriminator_is_rejected(self):
        """CudaGraphConfig.mode has a default, which is what makes omitting it
        look safe; the union needs it as a tag and the worker dies without."""
        found = problems("cuda_graph_config:\n  max_batch_size: 64")
        self.assertTrue(found)
        self.assertIn("discriminator", found[0])

    def test_a_misspelt_value_in_a_str_field_is_rejected(self):
        """attn_backend is typed `str`, so nothing else catches FLASHINFR."""
        found = problems("attn_backend: FLASHINFR")
        self.assertTrue(found)
        self.assertIn("FLASHINFER", found[0])

    def test_a_backend_whose_library_is_absent_is_rejected(self):
        found = problems("moe_config:\n  backend: DEEPGEMM")
        self.assertTrue(found)
        self.assertIn("deep_gemm", found[0])

    def test_a_backend_whose_library_is_present_is_accepted(self):
        self.assertEqual(problems("attn_backend: FLASHINFER"), [])

    def test_the_full_ab_surface_renders_and_validates(self):
        """Every axis at its default, as the script writes it."""
        self.assertEqual(problems("""
attn_backend: TRTLLM
allreduce_strategy: AUTO
enable_chunked_prefill: false
moe_config:
  backend: AUTO
cuda_graph_config:
  mode: decode
  max_batch_size: 64
  enable_padding: true
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 4096
"""), [])

    def test_a_misspelt_allreduce_strategy_is_rejected(self):
        """This one IS a Literal, so the type check catches it without an
        allowlist -- unlike attn_backend, which is a bare str."""
        found = problems("allreduce_strategy: LOWPRECISON")
        self.assertTrue(found)
        self.assertIn("allreduce_strategy", found[0])

    def test_every_allreduce_strategy_the_engine_lists_is_accepted(self):
        for s in ("AUTO", "NCCL", "UB", "MINLATENCY", "ONESHOT", "TWOSHOT",
                  "LOWPRECISION", "MNNVL", "NCCL_SYMMETRIC"):
            self.assertEqual(problems(f"allreduce_strategy: {s}"), [], s)
