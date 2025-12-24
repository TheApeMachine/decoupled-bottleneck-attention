import unittest
from pathlib import Path

from caramba.cli import CLI


class TestCarambaCLISubparsers(unittest.TestCase):
    """
    TestCarambaCLISubparsers validates that subparser construction is stable.
    """

    def test_compile_subcommand_parses(self) -> None:
        """
        test_compile_subcommand_parses ensures the compile parser is usable.
        """
        cli = CLI()
        args = cli.parse_args(
            [
                "compile",
                "caramba/config/presets/llama32_1b_dba.yml",
                "--print-plan",
            ],
        )
        self.assertEqual(args.command, "compile")
        self.assertEqual(
            args.compile_manifest,
            Path("caramba/config/presets/llama32_1b_dba.yml"),
        )
        self.assertTrue(args.print_plan)


if __name__ == "__main__":
    unittest.main()


