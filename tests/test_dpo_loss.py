
import importlib.util
import math
import unittest
from types import SimpleNamespace

from src.dpo_loss import (
    completion_logps,
    dpo_loss,
    pair_inputs,
    pair_logps,
    scalar_loss,
)


class PairTests(unittest.TestCase):
    def test_shared_prompt_and_four_token_completion(self):
        row = {
            "prompt_token_ids": [8, 9],
            "chosen_token_ids": [1, 2, 3, 0],
            "rejected_token_ids": [1, 2, 4, 0],
        }
        self.assertEqual(
            pair_inputs(row), [[8, 9, 1, 2, 3, 0], [8, 9, 1, 2, 4, 0]]
        )

    def test_distinct_targets_required(self):
        row = {
            "prompt_token_ids": [8],
            "chosen_token_ids": [1, 2, 3, 0],
            "rejected_token_ids": [1, 2, 3, 0],
        }
        with self.assertRaises(ValueError):
            pair_inputs(row)

    def test_scalar_oracle_is_stable(self):
        self.assertAlmostEqual(scalar_loss(-4, -5, -4, -5, 0.5), math.log(2))
        self.assertEqual(scalar_loss(10000, 0, 0, 0, 1), 0)
        self.assertEqual(scalar_loss(-10000, 0, 0, 0, 1), 10000)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TensorTests(unittest.TestCase):
    def test_full_vocabulary_and_sum_over_four_tokens(self):
        import torch

        logits = torch.zeros((2, 4, 11))
        targets = torch.tensor([[1, 2, 3, 0], [1, 2, 4, 0]])
        torch.testing.assert_close(
            completion_logps(logits, targets), torch.full((2,), -4 * math.log(11))
        )

    def test_reference_detached_and_scalar_agrees(self):
        import torch

        policy = torch.tensor([-4.0, -5.0], requires_grad=True)
        reference = torch.tensor([-3.0, -6.0], requires_grad=True)
        loss = dpo_loss(policy, reference, 0.5)
        self.assertAlmostEqual(
            loss.item(), scalar_loss(-4, -5, -3, -6, 0.5), places=6
        )
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertLess(policy.grad[0].item(), 0)
        self.assertGreater(policy.grad[1].item(), 0)

    def test_only_completion_prediction_positions_receive_gradient(self):
        import torch

        ids = torch.tensor([[8, 9, 1, 2, 3, 0], [8, 9, 1, 2, 4, 0]])
        logits = torch.zeros((2, 6, 11), requires_grad=True)
        def model(**kwargs):
            return SimpleNamespace(logits=logits)

        values = pair_logps(model, ids, ids[:, -4:], full_logits=True)
        values.sum().backward()
        self.assertEqual(torch.count_nonzero(logits.grad[:, 0]).item(), 0)
        self.assertEqual(torch.count_nonzero(logits.grad[:, -1]).item(), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[:, 1:5]).item(), 0)

    def test_reduced_and_full_logits_agree(self):
        import torch

        ids = torch.tensor([[8, 9, 1, 2, 3, 0], [8, 9, 1, 2, 4, 0]])
        logits = torch.arange(132, dtype=torch.float32).reshape(2, 6, 11) / 100

        def model(**kwargs):
            selected = (
                logits[:, -kwargs["logits_to_keep"]:]
                if "logits_to_keep" in kwargs
                else logits
            )
            return SimpleNamespace(logits=selected)

        torch.testing.assert_close(
            pair_logps(model, ids, ids[:, -4:]),
            pair_logps(model, ids, ids[:, -4:], full_logits=True),
        )


if __name__ == "__main__":
    unittest.main()
