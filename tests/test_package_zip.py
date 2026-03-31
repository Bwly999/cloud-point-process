import tempfile
import unittest
import zipfile
from pathlib import Path

from package_zip import build_default_output_path, package_project


class TestPackageZip(unittest.TestCase):
    def test_package_project_includes_only_code_and_docs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "demo-project"
            root.mkdir()

            (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / "process_heightmap.py").write_text("print('demo')\n", encoding="utf-8")

            package_dir = root / "cloud_point_process"
            package_dir.mkdir()
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "processor.py").write_text("VALUE = 1\n", encoding="utf-8")
            pycache_dir = package_dir / "__pycache__"
            pycache_dir.mkdir()
            (pycache_dir / "processor.cpython-37.pyc").write_bytes(b"pyc")

            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_processor.py").write_text("def test_ok(): pass\n", encoding="utf-8")

            docs_dir = root / "docs" / "superpowers" / "specs"
            docs_dir.mkdir(parents=True)
            (docs_dir / "design.md").write_text("spec\n", encoding="utf-8")

            demo_outputs = root / "demo_outputs"
            demo_outputs.mkdir()
            (demo_outputs / "overview.png").write_bytes(b"png")

            (root / "cloud-point-process.zip").write_bytes(b"old")

            output_path = root / "release.zip"
            package_project(root, output_path)

            with zipfile.ZipFile(output_path) as archive:
                names = sorted(archive.namelist())

            self.assertIn(".gitignore", names)
            self.assertIn("README.md", names)
            self.assertIn("process_heightmap.py", names)
            self.assertIn("cloud_point_process/__init__.py", names)
            self.assertIn("cloud_point_process/processor.py", names)
            self.assertIn("tests/test_processor.py", names)
            self.assertIn("docs/superpowers/specs/design.md", names)

            self.assertNotIn("demo_outputs/overview.png", names)
            self.assertNotIn("cloud_point_process/__pycache__/processor.cpython-37.pyc", names)
            self.assertNotIn("cloud-point-process.zip", names)

    def test_build_default_output_path_uses_repo_name(self):
        root = Path(r"D:\work\cloud-point-process")
        output = build_default_output_path(root)
        self.assertEqual(output, root / "cloud-point-process-package.zip")


if __name__ == "__main__":
    unittest.main()
