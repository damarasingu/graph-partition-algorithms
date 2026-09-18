import unittest

from graph_partitioning import (
    BayesianMarkovBoundaryModel,
    dynamic_program_partition,
    exhaustive_partition,
    generate_instance,
    objective,
)


class GraphPartitioningTests(unittest.TestCase):
    def test_dynamic_program_matches_reference_solver(self):
        weights = generate_instance(9, seed=11)
        exact = exhaustive_partition(weights, 3)
        optimized = dynamic_program_partition(weights, 3, BayesianMarkovBoundaryModel(smoothness=0.0))
        self.assertAlmostEqual(exact.cost, optimized.cost)

    def test_partition_is_contiguous_and_covers_path(self):
        weights = [1.0, 4.0, 2.0, 5.0, 3.0]
        result = dynamic_program_partition(weights, 2)
        self.assertEqual(len(result.cuts), 1)
        self.assertEqual(result.cuts[0], 3)
        self.assertAlmostEqual(objective(weights, result.cuts), 0.5)

    def test_model_detects_large_local_discontinuity(self):
        model = BayesianMarkovBoundaryModel()
        probabilities = model.cut_probabilities([1.0, 1.0, 8.0, 8.0])
        self.assertGreater(probabilities[1], probabilities[0])
        self.assertGreater(probabilities[1], probabilities[2])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            dynamic_program_partition([], 1)
        with self.assertRaises(ValueError):
            dynamic_program_partition([1.0, 2.0], 3)
        with self.assertRaises(ValueError):
            dynamic_program_partition([1.0, -2.0], 1)


if __name__ == "__main__":
    unittest.main()
