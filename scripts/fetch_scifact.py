"""Download the official SciFact release and extract the files used here.

The tarball is verified against ``data.source_sha256`` from config.yaml before
anything is extracted. Only the corpus and claim files needed by the study are
written to ``data/raw/scifact``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

from common import DataError, load_config, resolve, sha256_bytes, sha256_file

WANTED_FILES = ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl")


def download_and_verify(url: str, dest: Path, expected_sha256: str) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise DataError(
            f"download checksum mismatch: expected {expected_sha256}, got {actual}"
        )
    dest.write_bytes(payload)
    return dest


def extract_dataset(tar_path: Path, dest_dir: Path, names: tuple[str, ...]) -> list[Path]:
    """Extract selected basenames from the release tarball into dest_dir."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(tar_path, "r:gz") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            basename = Path(member.name).name
            if basename in names:
                if not member.isfile():
                    raise DataError(f"tarball member {member.name!r} is not a file")
                if basename in members:
                    raise DataError(f"tarball contains duplicate {basename!r}")
                members[basename] = member
        missing = [name for name in names if name not in members]
        if missing:
            raise DataError(f"tarball is missing files: {missing}")
        for name in names:
            handle = archive.extractfile(members[name])
            if handle is None:  # pragma: no cover - guarded by isfile above
                raise DataError(f"cannot read tarball member {name!r}")
            target = dest_dir / name
            target.write_bytes(handle.read())
            extracted.append(target)
    return extracted


def fetch(config_path: Path, force: bool = False) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = load_config(config_path)
    raw_dir = resolve(root, config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    tarball = raw_dir / "scifact.tar.gz"
    expected = config["data"]["source_sha256"]

    if force or not tarball.is_file() or sha256_file(tarball) != expected:
        download_and_verify(config["data"]["source_url"], tarball, expected)

    extracted: list[str] = []
    if force or not all((raw_dir / name).is_file() for name in WANTED_FILES):
        extracted = [
            path.name for path in extract_dataset(tarball, raw_dir, WANTED_FILES)
        ]

    return {
        "tarball": str(tarball.relative_to(root)),
        "tarball_sha256": sha256_file(tarball),
        "files": sorted(
            str(path.relative_to(root))
            for path in raw_dir.iterdir()
            if path.is_file()
        ),
        "extracted_now": extracted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    args = parser.parse_args(argv)
    summary = fetch(args.config, force=args.force)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
