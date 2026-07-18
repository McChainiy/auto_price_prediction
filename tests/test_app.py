from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTest(unittest.TestCase):
    def test_light_theme_is_configured(self) -> None:
        with (PROJECT_ROOT / ".streamlit" / "config.toml").open("rb") as config_file:
            theme = tomllib.load(config_file)["theme"]

        self.assertEqual(theme["base"], "light")
        self.assertEqual(theme["backgroundColor"], "#FFFFFF")
        self.assertEqual(theme["textColor"], "#172033")

    def test_app_starts_with_primary_workflow_first(self) -> None:
        app = AppTest.from_file(
            PROJECT_ROOT / "app.py",
            default_timeout=30,
        ).run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 5)
        self.assertEqual(app.tabs[0].label, "Estimate a car")


if __name__ == "__main__":
    unittest.main()
