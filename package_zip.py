from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Iterable, List


INCLUDE_PATHS = [
    ".gitignore",
    "README.md",
    "process_heightmap.py",
    "package_zip.py",
    "cloud_point_process",
    "tests",
    "docs",
]

SKIP_DIR_NAMES = {"__pycache__"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def iter_package_files(root: Path) -> Iterable[Path]:
    for relative in INCLUDE_PATHS:
        path = root / relative
        if not path.exists():
            continue

        if path.is_file():
            yield path
            continue

        for child in sorted(path.rglob("*")):
            if child.is_dir():
                continue
            if any(part in SKIP_DIR_NAMES for part in child.relative_to(root).parts):
                continue
            if child.suffix.lower() in SKIP_SUFFIXES:
                continue
            yield child


def build_default_output_path(root: Path) -> Path:
    return root / "{}-package.zip".format(root.name)


def package_project(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    seen = set()
    for path in iter_package_files(root):
        resolved = path.resolve()
        if resolved == output_path:
            continue
        rel = resolved.relative_to(root)
        if rel in seen:
            continue
        seen.add(rel)
        files.append(resolved)

    with zipfile.ZipFile(str(output_path), mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(str(path), arcname=str(path.relative_to(root)))

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将仓库中的代码和文档打包为 zip，不包含样例输出和临时文件。")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 zip 路径。默认输出到仓库根目录，文件名为 <仓库名>-package.zip。",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="要打包的仓库根目录。默认是当前脚本所在目录。",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    output_path = Path(args.output).resolve() if args.output else build_default_output_path(root)
    result = package_project(root, output_path)
    print("打包完成：{}".format(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
