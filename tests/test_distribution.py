import unittest
from unittest.mock import patch

import torch

from distribution import DiscretizedDistribution, Exponential


class TestDiscretizedExponential(unittest.TestCase):
    @staticmethod
    def make_distribution(rate: float) -> DiscretizedDistribution:
        base_distribution = (Exponential, {"rate": rate, "boundary": 1e-6})
        with patch(
            "distribution.return_class_and_args",
            return_value=base_distribution,
        ):
            return DiscretizedDistribution(base_distribution=None).double()

    def test_softmax_matches_normalized_cdf_bin_probabilities(self):
        cases = (
            (0.001, 1),
            (0.001, 32),
            (0.01, 256),
            (0.1, 64),
            (1.0, 16),
        )

        for rate, width in cases:
            with self.subTest(rate=rate, width=width):
                distribution = self.make_distribution(rate)
                x = torch.arange(width, dtype=torch.float64).unsqueeze(1)

                bin_probabilities = (
                    distribution.base_distribution.cdf(x + 1.0)
                    - distribution.base_distribution.cdf(x)
                ).squeeze(1)
                old_log_probabilities = torch.log(bin_probabilities)
                old_probabilities = old_log_probabilities.exp()
                old_probabilities = old_probabilities / old_probabilities.sum()

                softmax_probabilities = distribution.compute_probability_vector(x)

                torch.testing.assert_close(
                    softmax_probabilities,
                    old_probabilities,
                    rtol=1e-12,
                    atol=1e-15,
                )


if __name__ == "__main__":
    unittest.main()
