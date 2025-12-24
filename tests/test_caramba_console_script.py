import unittest
from pathlib import Path
import importlib
import sys
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = cast(Any, importlib.import_module("tomli"))


class TestCarambaConsoleScript(unittest.TestCase):
    def test_pyproject_exposes_caramba_console_script(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        pyproject = repo / "pyproject.toml"
        self.assertTrue(pyproject.exists())

        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        self.assertEqual(
            data.get("project", {}).get("scripts", {}).get("caramba"),
            "caramba.__main__:main",
        )

    def test_caramba_main_module_exists(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        entry = repo / "caramba" / "__main__.py"
        self.assertTrue(entry.exists())


if __name__ == "__main__":
    unittest.main()

