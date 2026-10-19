"""Build a deterministic source ZIP and verify every archived file by SHA256."""

import argparse
import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"__pycache__", "build", "dist", "data", "outputs", "checkpoints", ".venv", ".git", ".pytest_cache"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT.parent / "BRIDGE-code-release-0.1.0rc1.zip")
    args = parser.parse_args()
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink() or any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in {".pt", ".pth", ".pkl", ".h5", ".h5ad", ".npz", ".pyc", ".pem", ".key"} or path.name in {"SHA256SUMS", ".env"}:
            continue
        files.append(path)
    files.sort(key=lambda p: p.relative_to(ROOT).as_posix())
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest = "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    (ROOT / "SHA256SUMS").write_text(manifest)
    files.append(ROOT / "SHA256SUMS")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(ROOT.name + "/" + relative, date_time=(2026, 10, 17, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(ROOT.name + "/" + name)).hexdigest() != expected:
                raise RuntimeError("ZIP content checksum mismatch: " + name)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(f"{digest}  {args.out.name}\n")
    print(f"{args.out}\n{len(files)} files; {args.out.stat().st_size} bytes\nSHA256 {digest}")


if __name__ == "__main__":
    main()
