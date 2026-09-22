#!/usr/bin/env bash
# Verify and extract a STAR-VAE checkpoint archive downloaded in a browser.

set -euo pipefail

command -v python3 >/dev/null || { printf 'Error: Python 3 is required.\n' >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints}"
BUNDLE_MANIFEST="${REPO_ROOT}/checkpoint_bundle.json"
CHECKSUM_MANIFEST="${REPO_ROOT}/checkpoints.sha256"
ARCHIVE_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["archive"])' "${BUNDLE_MANIFEST}")"
ARCHIVE_PATH="${REPO_ROOT}/${ARCHIVE_NAME}"
ACCEPT_UNVERIFIED=false
BOX_URL="https://ibm.ent.box.com/v/star-vae-checkpoints"

usage() {
    printf '%s\n' \
        "Usage: bash scripts/install_checkpoints.sh [options]" \
        "" \
        "Options:" \
        "  --zip PATH                       Checkpoint ZIP (default: ${ARCHIVE_NAME})" \
        "  --dir PATH                       Destination checkpoint directory" \
        "  --accept-unverified-integrity    Allow an archive whose SHA-256 differs from the tested release" \
        "  -h, --help                       Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zip|--dir)
            [[ $# -ge 2 ]] || { printf 'Error: %s requires a value.\n' "$1" >&2; exit 2; }
            if [[ "$1" == "--zip" ]]; then ARCHIVE_PATH="$2"; else CHECKPOINT_DIR="$2"; fi
            shift 2
            ;;
        --accept-unverified-integrity) ACCEPT_UNVERIFIED=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

ARCHIVE_PATH="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))' "${ARCHIVE_PATH}")"
CHECKPOINT_DIR="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))' "${CHECKPOINT_DIR}")"

[[ -f "${ARCHIVE_PATH}" ]] || {
    printf 'Error: archive not found: %s\nDownload it manually from %s\n' "${ARCHIVE_PATH}" "${BOX_URL}" >&2
    exit 1
}
if [[ -d "${CHECKPOINT_DIR}" ]] && compgen -G "${CHECKPOINT_DIR}/*" >/dev/null; then
    printf 'Error: destination is not empty; refusing to overwrite: %s\n' "${CHECKPOINT_DIR}" >&2
    exit 1
fi
if [[ -e "${CHECKPOINT_DIR}" ]] && [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    printf 'Error: destination exists and is not a directory: %s\n' "${CHECKPOINT_DIR}" >&2
    exit 1
fi

destination_parent="$(dirname "${CHECKPOINT_DIR}")"
mkdir -p "${destination_parent}"
stage="$(mktemp -d "${destination_parent}/.star-vae-checkpoints.XXXXXX")"
cleanup() { rm -rf "${stage}"; }
trap cleanup EXIT

python3 - "${ARCHIVE_PATH}" "${BUNDLE_MANIFEST}" "${CHECKSUM_MANIFEST}" "${stage}" "${ACCEPT_UNVERIFIED}" <<'PY'
import hashlib
import json
import re
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

archive_path, manifest_path, checksum_path, stage_path, accept_unverified = sys.argv[1:]


def normalized(value, field, *, directory=False):
    if not isinstance(value, str) or not value:
        raise SystemExit(f"Error: {field} must be a non-empty string")
    if any(character in value for character in ("\\", "\0", "\n", "\r")):
        raise SystemExit(f"Error: unsafe {field}: {value!r}")
    plain = value[:-1] if directory and value.endswith("/") else value
    path = PurePosixPath(plain)
    if not plain or path.is_absolute() or plain.startswith("/"):
        raise SystemExit(f"Error: unsafe {field}: {value!r}")
    if any(part in ("", ".", "..") for part in plain.split("/")):
        raise SystemExit(f"Error: unsafe {field}: {value!r}")
    if path.parts and path.parts[0].endswith(":"):
        raise SystemExit(f"Error: unsafe {field}: {value!r}")
    if path.as_posix() != plain or (not directory and value.endswith("/")):
        raise SystemExit(f"Error: non-normalized {field}: {value!r}")
    return plain


def digest_stream(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
archive_name = normalized(manifest.get("archive"), "manifest archive")
if "/" in archive_name:
    raise SystemExit("Error: manifest archive must be a file name")
root = normalized(manifest.get("root"), "manifest root")
raw_members = [entry["checkpoint"] for entry in manifest["models"].values()]
raw_members.extend(manifest.get("additional_files", []))
members = [normalized(member, "manifest member") for member in raw_members]
duplicates = sorted({member for member in members if members.count(member) > 1})
if duplicates:
    raise SystemExit(f"Error: duplicate manifest members: {duplicates}")
expected_files = {f"{root}/{member}" for member in members}
allowed_directories = {root}
for member in expected_files:
    parent = PurePosixPath(member).parent
    while parent.as_posix() != ".":
        allowed_directories.add(parent.as_posix())
        parent = parent.parent

checksums = {}
for line_number, line in enumerate(Path(checksum_path).read_text(encoding="utf-8").splitlines(), 1):
    match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
    if not match:
        raise SystemExit(f"Error: malformed checksum line {line_number}")
    checksum, name = match.groups()
    name = normalized(name, "checksum member")
    if name in checksums:
        raise SystemExit(f"Error: duplicate checksum member: {name}")
    checksums[name] = checksum.lower()
expected_checksum_names = expected_files | {archive_name}
if set(checksums) != expected_checksum_names:
    raise SystemExit(
        "Error: checksum membership differs from checkpoint_bundle.json; "
        f"missing={sorted(expected_checksum_names - set(checksums))}, "
        f"extra={sorted(set(checksums) - expected_checksum_names)}"
    )

with open(archive_path, "rb") as handle:
    archive_digest = digest_stream(handle)
if archive_digest != checksums[archive_name]:
    if accept_unverified != "true":
        raise SystemExit(
            "Error: checkpoint archive SHA-256 mismatch.\n"
            f"Expected: {checksums[archive_name]}\nActual:   {archive_digest}\n"
            "Refusing extraction. Use --accept-unverified-integrity only if you trust this different archive."
        )
    print(f"WARNING: accepting checkpoint archive with unrecognized SHA-256: {archive_digest}", file=sys.stderr)
else:
    print(f"Verified checkpoint archive SHA-256: {archive_digest}")

with zipfile.ZipFile(archive_path) as archive:
    infos = archive.infolist()
    if not infos:
        raise SystemExit("Error: archive is empty")
    files = {}
    seen = set()
    directories = set()
    for info in infos:
        is_directory = info.is_dir()
        name = normalized(info.filename, "archive member", directory=is_directory)
        if name in seen:
            raise SystemExit(f"Error: duplicate normalized archive member: {name}")
        seen.add(name)
        mode_type = stat.S_IFMT((info.external_attr >> 16) & 0xFFFF)
        if is_directory:
            if mode_type not in (0, stat.S_IFDIR) or info.file_size != 0:
                raise SystemExit(f"Error: unsupported archive member type: {info.filename}")
            directories.add(name)
        else:
            if mode_type not in (0, stat.S_IFREG) or info.flag_bits & 0x1:
                raise SystemExit(f"Error: unsupported archive member type: {info.filename}")
            files[name] = info
    if set(files) != expected_files or not directories <= allowed_directories:
        raise SystemExit(
            "Error: archive membership differs from checkpoint_bundle.json; "
            f"missing={sorted(expected_files - set(files))}, "
            f"extra={sorted((set(files) - expected_files) | (directories - allowed_directories))}"
        )

    for name, info in files.items():
        with archive.open(info) as handle:
            actual = digest_stream(handle)
        if actual != checksums[name]:
            raise SystemExit(f"Error: extracted checkpoint SHA-256 mismatch: {name[len(root) + 1:]}")

    payload = Path(stage_path) / root
    for name in sorted(files):
        relative = PurePosixPath(name).relative_to(root)
        target = payload.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(files[name]) as source, target.open("xb") as destination:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                destination.write(chunk)

print(archive_digest)
PY

payload="${stage}/$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["root"])' "${BUNDLE_MANIFEST}")"
if [[ -d "${CHECKPOINT_DIR}" ]]; then rmdir "${CHECKPOINT_DIR}"; fi
mv "${payload}" "${CHECKPOINT_DIR}"
trap - EXIT
rm -rf "${stage}"

printf 'Installed checkpoints in: %s\n' "${CHECKPOINT_DIR}"
printf '%s\n' \
    'The original ZIP was retained.' \
    'Only load these PyTorch checkpoints if you trust their source.'
