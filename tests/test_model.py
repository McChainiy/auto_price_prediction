from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.model import MyTransormer, load_pipeline, predict_prices, prepare_features


class ModelPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_pipeline()
        cls.data = pd.read_csv("test.csv")
        cls.target = cls.data["selling_price"]

    def test_predictions_match_versioned_artifact(self) -> None:
        predictions = predict_prices(self.model, self.data)
        expected_first_predictions = np.array(
            [
                335677.6949389104,
                906666.5284986442,
                532937.3183886298,
                608928.7547246290,
                83784.5828386506,
            ]
        )

        self.assertEqual(predictions.shape, (1000,))
        self.assertTrue(np.isfinite(predictions).all())
        np.testing.assert_allclose(
            predictions[:5],
            expected_first_predictions,
            rtol=0,
            atol=1e-6,
        )

    def test_documented_quality_metrics_are_reproducible(self) -> None:
        predictions = predict_prices(self.model, self.data)

        self.assertAlmostEqual(r2_score(self.target, predictions), 0.8869457592, places=9)
        self.assertAlmostEqual(
            mean_absolute_error(self.target, predictions),
            127521.59103679073,
            places=6,
        )

    def test_unknown_manufacturer_uses_safe_fallback(self) -> None:
        example = self.data.iloc[[0]].copy()
        example["name"] = "Newbrand Example"

        prediction = predict_prices(self.model, example)

        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction[0]))

    def test_output_can_be_clipped_to_price_domain(self) -> None:
        predictions = predict_prices(self.model, self.data, clip=True)
        self.assertTrue((predictions >= 0).all())

    def test_missing_columns_produce_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required columns: torque"):
            prepare_features(self.data.drop(columns="torque"))

    def test_measurement_parsing(self) -> None:
        transformer = MyTransormer()
        torque = transformer.extract_torque("12.5 kgm at 2500 rpm")
        rpm = transformer.extract_rpm("12.5 kgm at 2500 rpm")

        self.assertAlmostEqual(torque, 122.5)
        self.assertEqual(rpm, 2500)


if __name__ == "__main__":
    unittest.main()
