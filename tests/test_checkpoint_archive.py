"""Cheap synthetic tests for checkpoint archive creation and installation."""

import hashlib
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_checkpoint_archive.py"
INSTALLER = ROOT / "scripts/install_checkpoints.sh"
MANIFEST = ROOT / "checkpoint_bundle.json"


def _expected_members():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = [entry["checkpoint"] for entry in manifest["models"].values()]
    paths.extend(manifest.get("additional_files", []))
    return manifest, sorted(paths)


def _manifest(*members):
    return {
        "archive": "fixture.zip",
        "root": "checkpoints",
        "models": {
            f"model-{index}": {"checkpoint": member, "fixture": True}
            for index, member in enumerate(members)
        },
        "additional_files": [],
    }


def _run_builder(tmp_path, manifest, *, force=False):
    source = tmp_path / "source"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "fixture.zip"
    checksums = tmp_path / "fixture.sha256"
    source.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    command = [
        sys.executable,
        str(BUILDER),
        "--source", str(source),
        "--manifest", str(manifest_path),
        "--output", str(output),
        "--checksums", str(checksums),
    ]
    if force:
        command.append("--force")
    return source, output, checksums, command


def _zip_info(name, mode=stat.S_IFREG | 0o644):
    info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = mode << 16
    return info


def _installer_project(tmp_path, entries, *, manifest=None):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy2(INSTALLER, project / "scripts/install_checkpoints.sh")
    manifest = manifest or _manifest("model/last.ckpt")
    (project / "checkpoint_bundle.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    archive = project / manifest["archive"]
    with ZipFile(archive, "w") as handle:
        for entry in entries:
            if len(entry) == 2:
                name, content = entry
                info = _zip_info(name)
            else:
                name, content, mode = entry
                info = _zip_info(name, mode)
            handle.writestr(info, content)

    expected_files = {
        f"{manifest['root']}/{entry['checkpoint']}"
        for entry in manifest["models"].values()
    }
    expected_files.update(
        f"{manifest['root']}/{name}"
        for name in manifest.get("additional_files", [])
    )
    content_by_name = {name: content for name, content, *_ in entries}
    lines = [f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}"]
    for name in sorted(expected_files):
        content = content_by_name.get(name, b"missing")
        lines.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
    (project / "checkpoints.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    destination = tmp_path / "installed"
    command = [
        "bash",
        str(project / "scripts/install_checkpoints.sh"),
        "--zip", str(archive),
        "--dir", str(destination),
    ]
    return project, archive, destination, command


def test_checkpoint_manifest_is_complete_and_excludes_unsupported_smiles_ablation():
    manifest, members = _expected_members()
    assert manifest["archive"] == "star_vae_checkpoints_v1.0.0.zip"
    assert len(manifest["models"]) == 20
    assert len(members) == len(set(members)) == 22
    assert "ablations/a1_smiles_tokenization/last.ckpt" not in members
    assert "vae_pubchem80m/vocab.json" in members


def test_checkpoint_builder_is_deterministic(tmp_path):
    manifest = _manifest("a/last.ckpt", "b/last.ckpt")
    outputs = []
    checksum_payloads = []
    for run in (1, 2):
        run_dir = tmp_path / str(run)
        run_dir.mkdir()
        source, output, checksums, command = _run_builder(run_dir, manifest)
        for member in ("a/last.ckpt", "b/last.ckpt"):
            path = source / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture-{member}".encode())
        subprocess.run(command, check=True)
        outputs.append(output.read_bytes())
        checksum_payloads.append(checksums.read_text(encoding="utf-8"))
        with ZipFile(output) as archive:
            assert [info.filename for info in archive.infolist()] == [
                "checkpoints/a/last.ckpt",
                "checkpoints/b/last.ckpt",
            ]
    assert outputs[0] == outputs[1]
    assert checksum_payloads[0] == checksum_payloads[1]


@pytest.mark.parametrize(
    "member",
    ["../escape.ckpt", "/absolute.ckpt", "dir\\file.ckpt", "a/./file.ckpt",
     "a//file.ckpt", "nul\0file.ckpt", "line\nfile.ckpt"],
)
def test_builder_rejects_unsafe_manifest_members(tmp_path, member):
    _, output, checksums, command = _run_builder(tmp_path, _manifest(member))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert not output.exists()
    assert not checksums.exists()


def test_builder_rejects_duplicate_manifest_members(tmp_path):
    manifest = _manifest("same.ckpt")
    manifest["additional_files"] = ["same.ckpt"]
    source, output, checksums, command = _run_builder(tmp_path, manifest)
    (source / "same.ckpt").write_bytes(b"fixture")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Duplicate manifest members" in result.stderr
    assert not output.exists()
    assert not checksums.exists()


def test_builder_rejects_symlink_input(tmp_path):
    source, output, checksums, command = _run_builder(
        tmp_path, _manifest("model/last.ckpt")
    )
    outside = tmp_path / "outside.ckpt"
    outside.write_bytes(b"fixture")
    (source / "model").mkdir()
    (source / "model/last.ckpt").symlink_to(outside)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert not output.exists()
    assert not checksums.exists()


def test_builder_refuses_overwrite_without_force(tmp_path):
    source, output, checksums, command = _run_builder(
        tmp_path, _manifest("model/last.ckpt")
    )
    (source / "model").mkdir()
    (source / "model/last.ckpt").write_bytes(b"new")
    output.write_bytes(b"keep archive")
    checksums.write_bytes(b"keep checksums")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert output.read_bytes() == b"keep archive"
    assert checksums.read_bytes() == b"keep checksums"


def test_builder_refuses_broken_symlink_output_without_force(tmp_path):
    source, output, checksums, command = _run_builder(
        tmp_path, _manifest("model/last.ckpt")
    )
    (source / "model").mkdir()
    (source / "model/last.ckpt").write_bytes(b"new")
    output.symlink_to(tmp_path / "missing-target")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert output.is_symlink()
    assert not checksums.exists()


def test_builder_force_replaces_outputs_after_validation(tmp_path):
    source, output, checksums, command = _run_builder(
        tmp_path, _manifest("model/last.ckpt"), force=True
    )
    (source / "model").mkdir()
    (source / "model/last.ckpt").write_bytes(b"new")
    output.write_bytes(b"old archive")
    checksums.write_bytes(b"old checksums")
    subprocess.run(command, check=True)
    assert output.read_bytes() != b"old archive"
    assert "checkpoints/model/last.ckpt" in checksums.read_text(encoding="utf-8")


def test_installer_accepts_expected_directory_records(tmp_path):
    _, _, destination, command = _installer_project(
        tmp_path,
        [
            ("checkpoints/", b"", stat.S_IFDIR | 0o755),
            ("checkpoints/model/", b"", stat.S_IFDIR | 0o755),
            ("checkpoints/model/last.ckpt", b"fixture"),
        ],
    )
    subprocess.run(command, check=True)
    assert (destination / "model/last.ckpt").read_bytes() == b"fixture"


@pytest.mark.parametrize(
    "extra",
    [
        ("checkpoints/extra.ckpt", b"extra"),
        ("checkpoints/unexpected/", b"", stat.S_IFDIR | 0o755),
        ("../escape.ckpt", b"escape"),
        ("/absolute.ckpt", b"absolute"),
        ("checkpoints\\model\\last.ckpt", b"backslash"),
    ],
)
def test_installer_rejects_extra_or_unsafe_members(tmp_path, extra):
    _, _, destination, command = _installer_project(
        tmp_path, [("checkpoints/model/last.ckpt", b"fixture"), extra]
    )
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert not destination.exists()


def test_installer_rejects_duplicate_members(tmp_path):
    _, _, destination, command = _installer_project(
        tmp_path,
        [
            ("checkpoints/model/last.ckpt", b"first"),
            ("checkpoints/model/last.ckpt", b"second"),
        ],
    )
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "duplicate normalized archive member" in result.stderr
    assert not destination.exists()


def test_installer_rejects_symlink_member(tmp_path):
    _, _, destination, command = _installer_project(
        tmp_path,
        [("checkpoints/model/last.ckpt", b"../../escape", stat.S_IFLNK | 0o777)],
    )
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "unsupported archive member type" in result.stderr
    assert not destination.exists()


def test_installer_rejects_checksum_membership_not_derived_from_manifest(tmp_path):
    project, _, destination, command = _installer_project(
        tmp_path, [("checkpoints/model/last.ckpt", b"fixture")]
    )
    with (project / "checkpoints.sha256").open("a", encoding="utf-8") as handle:
        handle.write(f"{'0' * 64}  checkpoints/unlisted.ckpt\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "checksum membership differs" in result.stderr
    assert not destination.exists()
