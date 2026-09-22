"""
exp18_drugdiff_download_checkpoints.py
========================================
Download only the two required DrugDiff checkpoint files from Zenodo
without downloading the full 2.8 GB DrugDiff.zip archive.

Uses HTTP range requests to stream and decompress only:
  - DrugDiff/model/drugdiff.ckpt  (282 MB uncompressed): latent diffusion model
  - DrugDiff/model/sa.ckpt        (105 MB uncompressed): pre-trained SA Score predictor

Output:
  paper_results/exp18_conditional_generation_sascore/drugdiff/model/drugdiff.ckpt
  paper_results/exp18_conditional_generation_sascore/drugdiff/model/sa.ckpt

Zenodo record: doi:10.5281/zenodo.14772460

Usage:
    # The cited source does not publish authoritative per-file SHA-256 values.
    # The downloader therefore fails closed unless this risk is accepted:
    python bmfm_sm_gen/scripts/exp18_drugdiff_download_checkpoints.py \
        --accept-unverified-integrity
    python bmfm_sm_gen/scripts/exp18_drugdiff_download_checkpoints.py \
        --accept-unverified-integrity --force
"""

import argparse
import hashlib
import struct
import urllib.request
import zlib
from pathlib import Path

ZENODO_URL = "https://zenodo.org/api/records/14772460/files/DrugDiff.zip/content"

OUT_DIR = Path(__file__).resolve().parents[2] / \
          "paper_results/exp18_conditional_generation_sascore/drugdiff/model"

# Exact offsets determined from the ZIP central directory.
# Verified against Zenodo record 14772460 (DrugDiff v1).
FILES = [
    {
        "name":             "DrugDiff/model/drugdiff.ckpt",
        "local_hdr_offset": 791515408,
        "compress_size":    260356093,   # exact bytes in zip
        "fname_len":        26,
        "extra_len":        32,          # verified from local header
        "out_name":         "drugdiff.ckpt",
        "expected_size":    282334232,   # uncompressed (verified)
        "sha256":           None,        # not found in inspected source metadata
    },
    {
        "name":             "DrugDiff/model/sa.ckpt",
        "local_hdr_offset": 1207482805,
        "compress_size":    51517225,
        "fname_len":        22,
        "extra_len":        32,          # verified from local header
        "out_name":         "sa.ckpt",
        "expected_size":    105356698,
        "sha256":           None,        # not found in inspected source metadata
    },
]

CHUNK = 8 * 1024 * 1024  # 8 MB per HTTP range request


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_integrity(entry: dict, path: Path, accept_unverified: bool) -> str:
    expected = entry.get("sha256")
    actual = sha256_file(path)
    if expected and actual.lower() != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path}: got {actual}, expected {expected}"
        )
    if not expected and not accept_unverified:
        raise RuntimeError(
            f"No authoritative SHA-256 is recorded for {entry['name']}. "
            "Refusing to trust the checkpoint. Obtain and record a checksum "
            "from an authoritative source, or explicitly pass "
            "--accept-unverified-integrity to accept this risk."
        )
    return actual


def fetch_range(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{start + length - 1}"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def get_extra_len(url: str, local_hdr_offset: int) -> int:
    """Read the local file header to get the actual extra_len."""
    hdr = fetch_range(url, local_hdr_offset, 30)
    if hdr[:4] != b"PK\x03\x04":
        raise ValueError(f"Bad local file header at offset {local_hdr_offset}: {hdr[:4].hex()}")
    return struct.unpack_from("<H", hdr, 28)[0]


def download_and_extract(
    entry: dict, out_path: Path, force: bool, accept_unverified: bool
) -> None:
    if out_path.exists() and not force:
        size = out_path.stat().st_size
        if size == entry["expected_size"]:
            digest = verify_integrity(entry, out_path, accept_unverified)
            status = "verified" if entry.get("sha256") else "UNVERIFIED (explicitly accepted)"
            print(f"  Already exists ({size / 1e6:.0f} MB): {out_path}, {status}")
            print(f"  SHA-256: {digest}")
            return
        print(f"  Incomplete file ({size / 1e6:.0f} MB, expected "
              f"{entry['expected_size'] / 1e6:.0f} MB), re-downloading")

    # Verify / read actual extra_len from local header
    actual_extra = get_extra_len(ZENODO_URL, entry["local_hdr_offset"])
    data_offset = (entry["local_hdr_offset"] + 30
                   + entry["fname_len"] + actual_extra)

    compress_size = entry["compress_size"]
    print(f"  Downloading {compress_size / 1e6:.1f} MB compressed ...")

    compressed = bytearray()
    pos = data_offset
    while len(compressed) < compress_size:
        to_fetch = min(CHUNK, compress_size - len(compressed))
        chunk = fetch_range(ZENODO_URL, pos, to_fetch)
        compressed.extend(chunk)
        pos += len(chunk)
        pct = len(compressed) / compress_size * 100
        print(f"    {len(compressed) / 1e6:.0f} / {compress_size / 1e6:.0f} MB"
              f"  ({pct:.0f}%)", flush=True)

    print(f"  Decompressing ...")
    raw = zlib.decompress(bytes(compressed), -15)   # raw deflate (wbits=-15)

    if len(raw) != entry["expected_size"]:
        raise ValueError(
            f"Size mismatch after decompression: got {len(raw)}, "
            f"expected {entry['expected_size']}"
        )

    actual_sha256 = sha256_bytes(raw)
    expected_sha256 = entry.get("sha256")
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch after decompression: got {actual_sha256}, "
            f"expected {expected_sha256}"
        )

    out_path.write_bytes(raw)
    print(f"  Saved {len(raw) / 1e6:.1f} MB → {out_path}")
    print(f"  SHA-256: {actual_sha256}")
    if not expected_sha256:
        print("  Integrity status: UNVERIFIED (explicitly accepted)")


def main():
    parser = argparse.ArgumentParser(
        description="Download DrugDiff checkpoints from Zenodo (sparse extraction)"
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    parser.add_argument(
        "--accept-unverified-integrity",
        action="store_true",
        help="Allow checkpoints lacking an authoritative SHA-256 (unsafe; explicit opt-in)",
    )
    args = parser.parse_args()

    unknown_hashes = [entry["name"] for entry in FILES if not entry.get("sha256")]
    if unknown_hashes and not args.accept_unverified_integrity:
        parser.error(
            "authoritative SHA-256 values are not recorded for: "
            + ", ".join(unknown_hashes)
            + "; refusing network access by default. Pass "
              "--accept-unverified-integrity only after independently assessing the source."
        )

    print(
        "WARNING: PyTorch .ckpt files may execute arbitrary code during "
        "deserialization. Load only artifacts from a source you trust."
    )
    if unknown_hashes:
        print(
            "WARNING: No authoritative per-file SHA-256 is available in the "
            "cited source; integrity verification has been explicitly waived."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for entry in FILES:
        out_path = OUT_DIR / entry["out_name"]
        print(f"\n=== {entry['name']} ===")
        download_and_extract(
            entry,
            out_path,
            force=args.force,
            accept_unverified=args.accept_unverified_integrity,
        )

    print("\n=== All DrugDiff checkpoints ready ===")
    for entry in FILES:
        p = OUT_DIR / entry["out_name"]
        print(f"  {p.name:20s}  {p.stat().st_size / 1e6:.0f} MB  {p}")


if __name__ == "__main__":
    main()
