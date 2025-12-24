import unittest
from pathlib import Path


class TestCarambaConsoleScript(unittest.TestCase):
    def test_pyproject_exposes_caramba_console_script(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        pyproject = repo / "pyproject.toml"
        self.assertTrue(pyproject.exists())

        content = pyproject.read_text(encoding="utf-8")
        self.assertIn('[project.scripts]', content)
        self.assertIn('caramba = "caramba.__main__:main"', content)

    def test_caramba_main_module_exists(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        entry = repo / "caramba" / "__main__.py"
        self.assertTrue(entry.exists())


if __name__ == "__main__":
    unittest.main()

