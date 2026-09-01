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

    def test_cuda_graph_config_without_its_discriminator_is_fine(self):
        """A correction, kept as a test so the wrong version cannot return.

        Validating the field's annotation with TypeAdapter rejects a dict
        without `mode` -- "Unable to extract tag using discriminator 'mode'" --
        and that is what this test used to assert. The engine does not do that.
        TorchLlmArgs has a validator, infer_cuda_graph_config_mode, that fills
        `mode` in before the union is discriminated, so the dict resolves to
        DecodeCudaGraphConfig and the worker is fine.

        The lesson is the one this whole validator exists for: checking the
        annotation is not checking what the engine does. The check is now a
        real TorchLlmArgs construction."""
        self.assertEqual(problems("cuda_graph_config:\n  max_batch_size: 64"), [])

    def test_a_misspelt_value_in_a_str_field_is_rejected(self):
        """attn_backend is typed `str`, so nothing else catches FLASHINFR."""
        found = problems("attn_backend: FLASHINFR")
        self.assertTrue(found)
        self.assertIn("FLASHINFER", found[0])

    def test_a_backend_whose_library_is_absent_is_rejected(self):
        """WIDEEP genuinely needs deep_ep, which is not in this container."""
        found = problems("moe_config:\n  backend: WIDEEP")
        self.assertTrue(found)
        self.assertIn("deep_ep", found[0])

    def test_deepgemm_import_gate_was_the_wrong_question(self):
        """This test asserted the opposite twice, and the second time was worse.

        First it gated DEEPGEMM on `import deep_gemm`, which fails: TRT-LLM
        bundles its own `tensorrt_llm.deep_gemm`. So the gate was corrected to
        the bundled module and a test was written asserting DEEPGEMM validates
        -- this test. Both were answering "is the library present" when the
        constraint is "does this GPU run these kernels", and TRT-LLM's
        DeepGemmFusedMoE refuses anything but SM100/103.

        The bundled module still imports on an H200. It is the backend that
        refuses, at construction, inside the allocation. Import success is not
        support, and a gate that only asks about imports will keep passing
        things that cannot run.
        """
        import importlib

        importlib.import_module("tensorrt_llm.deep_gemm")

        p = problems("moe_config:\n  backend: DEEPGEMM")
        self.assertTrue(
            p, "the module imports, so only a hardware check can reject this"
        )
        self.assertNotIn(
            "not installed", " ".join(p), "it is installed; that was never the issue"
        )

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

    def test_every_mfu_knob_validates(self):
        """One knob per line of the MFU decomposition; all must reach the
        engine, since a knob that is silently ignored is the failure this file
        exists to prevent."""
        for body in (
            "enable_low_latency_host_dispatch: true",
            "torch_compile_config: {}",
            "torch_compile_config:\n  enable_inductor: true",
            "context_parallel_size: 2",
            "allreduce_strategy: LOWPRECISION",
        ):
            self.assertEqual(problems(body), [], body)

    def test_a_backend_this_gpu_cannot_run_is_refused(self):
        """DEEPGEMM was recommended in this repo for an H200 on the strength of
        a neighbouring stack's deep_gemm result. That stack was vLLM, whose
        DeepGEMM is not TRT-LLM's DeepGemmFusedMoE, and TRT-LLM's refuses
        anything but SM100/103. The import gate passed it because
        tensorrt_llm.deep_gemm ships regardless. Import success is not
        support."""
        p = problems("moe_config:\n  backend: DEEPGEMM")
        self.assertTrue(p, "DEEPGEMM must not pass unchecked")
        joined = " ".join(p)
        self.assertIn("SM100", joined, joined)

    def test_an_unavailable_gpu_is_reported_as_unchecked_not_as_supported(self):
        """A validator that passes on a login node because it cannot see a GPU
        is worse than one that does not check: it converts 'unknown' into
        'fine'."""
        p = problems("moe_config:\n  backend: TRTLLM")
        self.assertTrue(p)
        self.assertRegex(" ".join(p), r"NOT checked|SM\d+")

    def test_the_hopper_default_passes(self):
        """CUTLASS is the only MoE backend an H200 with an fp8 checkpoint can
        run, and AUTO resolves to it. Both must validate everywhere."""
        self.assertEqual(problems("moe_config:\n  backend: CUTLASS"), [])
        self.assertEqual(problems("moe_config:\n  backend: AUTO"), [])
