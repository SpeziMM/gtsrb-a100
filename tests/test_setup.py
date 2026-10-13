import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SetupTests(unittest.TestCase):
    def test_old_python_gets_guidance_before_data_import(self):
        # Run the real package entry point with the reported version substituted.
        code = (
            "import sys, runpy; sys.version_info=(3,6,0,'final',0); "
            "runpy.run_module('gtsrb.data', run_name='__main__')"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires Python 3.10", result.stderr)
        self.assertIn("Python 3.6", result.stderr)
        self.assertNotIn("SyntaxError", result.stderr)
        # The early check itself must parse on the old interpreter.
        ast.parse((ROOT/"gtsrb/__init__.py").read_text(), feature_version=(3, 6))

    def test_explicit_supported_interpreter_check(self):
        result = subprocess.run(["bash", "scripts/setup.sh", "--check"], cwd=ROOT,
                                env=dict(os.environ, PYTHON_BIN=sys.executable),
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(sys.executable, result.stdout)

    def test_legacy_environment_is_preserved_and_setup_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT/"scripts", root/"scripts")
            (root/".venv/bin").mkdir(parents=True)
            python = root/".venv/bin/python"
            python.write_text("#!/bin/sh\nexit 1\n")
            python.chmod(0o755)
            result = subprocess.run(["bash", "scripts/setup.sh"], cwd=root,
                                    env=dict(os.environ, PYTHON_BIN=sys.executable),
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mv .venv .venv-python-old", result.stderr)
            self.assertEqual(python.read_text(), "#!/bin/sh\nexit 1\n")


if __name__ == "__main__":
    unittest.main()
