"""Tests of the near-tie rule of verify_all.py (`python3 -m unittest scripts/colab/test_near_tie.py`, no GPU needed).

The values are the ones measured on an L4 (Llama-3.2-3B Q4_K_M, "The capital of France is", the sixth generated token):
the CUDA run put ' Italy' 0.107 nats above ' France', the CPU run put ' France' 0.059 above ' Italy'.
"""
import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location("verify_all", os.path.join(os.path.dirname(__file__), "verify_all.py"))
va = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(va)

CUDA = {"tokens": ["a", "b", " Italy"], "tops": [{}, {}, {" Italy": -1.7179, " France": -1.8245}]}
CPU = {"tokens": ["a", "b", " France"], "tops": [{}, {}, {" France": -1.7257, " Italy": -1.7851}]}


class NearTie(unittest.TestCase):
    def test_the_measured_case_is_a_near_tie(self):
        self.assertIn("near tie", va.near_tie(CUDA, CPU))

    def test_identical_tokens_are_not_a_divergence(self):
        self.assertIsNone(va.near_tie(CUDA, CUDA))

    def test_a_clear_winner_on_the_cpu_side_is_a_defect(self):
        far = {"tokens": CPU["tokens"], "tops": [{}, {}, {" France": -0.1, " Italy": -3.0}]}
        self.assertIsNone(va.near_tie(CUDA, far))

    def test_a_clear_winner_on_the_gpu_side_is_a_defect(self):
        far = {"tokens": CUDA["tokens"], "tops": [{}, {}, {" Italy": -0.2, " France": -2.9}]}
        self.assertIsNone(va.near_tie(far, CPU))

    def test_a_token_outside_the_five_best_is_a_defect(self):
        gone = {"tokens": CPU["tokens"], "tops": [{}, {}, {" France": -1.7, " the": -1.8}]}
        self.assertIsNone(va.near_tie(CUDA, gone))

    def test_missing_details_are_not_a_near_tie(self):
        self.assertIsNone(va.near_tie({"tokens": ["a"], "tops": []}, {"tokens": ["b"], "tops": []}))

    def test_details_missing_on_one_side_only_are_not_a_near_tie(self):
        self.assertIsNone(va.near_tie(CUDA, {"tokens": CPU["tokens"], "tops": []}))
        self.assertIsNone(va.near_tie({"tokens": CUDA["tokens"], "tops": []}, CPU))


if __name__ == "__main__":
    unittest.main()
