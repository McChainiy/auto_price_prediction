from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTest(unittest.TestCase):
    def test_app_starts_with_primary_workflow_first(self) -> None:
        app = AppTest.from_file(
            PROJECT_ROOT / "app.py",
            default_timeout=30,
        ).run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 5)
        self.assertEqual(app.tabs[0].label, "Оценить автомобиль")


if __name__ == "__main__":
    unittest.main()
