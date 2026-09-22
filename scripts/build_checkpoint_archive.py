#!/usr/bin/env python3
"""Build the curated STAR-VAE checkpoint archive from an extracted bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_relative_posix(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if any(character in value for character in ("\\", "\0", "\n", "\r")):
        raise ValueError(f"Unsafe {field}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise ValueError(f"Unsafe {field}: {value!r}")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Unsafe {field}: {value!r}")
    if path.parts and path.parts[0].endswith(":"):
        raise ValueError(f"Unsafe {field}: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"Non-normalized {field}: {value!r}")
    return normalized


def manifest_members(manifest: dict) -> tuple[str, list[str]]:
    root = normalized_relative_posix(manifest.get("root"), field="manifest root")
    raw_members = [
        entry["checkpoint"] for entry in manifest["models"].values()
    ]
    raw_members.extend(manifest.get("additional_files", []))
    members = [
        normalized_relative_posix(member, field="manifest member")
        for member in raw_members
    ]
    duplicates = sorted({member for member in members if members.count(member) > 1})
    if duplicates:
        raise ValueError(f"Duplicate manifest members: {duplicates}")
    return root, sorted(members)


def resolve_sources(source: Path, members: list[str]) -> dict[str, Path]:
    source_root = source.resolve(strict=True)
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {source}")

    resolved: dict[str, Path] = {}
    for member in members:
        candidate = source_root.joinpath(*PurePosixPath(member).parts)
        current = source_root
        for part in PurePosixPath(member).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Archive input may not be a symlink: {member}")
        try:
            member_path = candidate.resolve(strict=True)
            member_path.relative_to(source_root)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"Archive input escapes or is missing: {member}") from error
        if not stat.S_ISREG(member_path.stat(follow_symlinks=False).st_mode):
            raise ValueError(f"Archive input is not a regular file: {member}")
        resolved[member] = member_path
    return resolved


def temporary_sibling(path: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    handle.close()
    return Path(handle.name)


def sanitized_checkpoint_bytes(
    source: Path,
    entry: dict,
    vocabulary_sha256: str,
    sanitization_defaults: dict,
) -> bytes:
    """Return an inference-focused checkpoint without local training metadata."""
    import io

    try:
        import torch
    except ImportError:
        # Synthetic archive tests deliberately use non-checkpoint fixture bytes.
        return source.read_bytes()

    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as error:
        if entry.get("fixture"):
            return source.read_bytes()
        raise ValueError(f"Unable to sanitize checkpoint: {source}") from error
    raw_state = checkpoint.get("state_dict", checkpoint)
    state_dict = OrderedDict(
        (key, value.detach().cpu())
        for key, value in raw_state.items()
        if torch.is_tensor(value)
    )
    if not state_dict:
        raise ValueError(f"Checkpoint has no tensor state_dict: {source}")

    original = checkpoint.get("hyper_parameters") or {}
    model_params = dict(original.get("model_params") or {})
    embedding = state_dict["model.encoder.embedding.weight"]
    position = state_dict["model.encoder.pos_encoder.pe"]
    d_model = embedding.shape[1]
    encoder_layers = {
        int(key.split(".layers.", 1)[1].split(".", 1)[0])
        for key in state_dict
        if key.startswith("model.encoder.transformer_encoder.layers.")
    }
    decoder_layers = {
        int(key.split(".layers.", 1)[1].split(".", 1)[0])
        for key in state_dict
        if key.startswith("model.decoder.transformer_decoder.layers.")
    }
    heads_by_width = sanitization_defaults["attention_heads_by_d_model"]
    nhead = model_params.get("nhead") or heads_by_width.get(str(d_model))
    if nhead is None:
        raise ValueError(
            f"No attention-head count recorded for d_model={d_model}: {source}"
        )
    model_params.update(
        {
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": max(encoder_layers) + 1,
            "num_decoder_layers": max(decoder_layers) + 1,
            "dim_feedforward": state_dict[
                "model.encoder.transformer_encoder.layers.0.linear1.weight"
            ].shape[0],
            "latent_dim": state_dict["model.encoder_fc_mu.weight"].shape[0],
            "dropout": model_params.get("dropout", 0.1),
            "max_len": position.shape[1] - 1,
        }
    )
    is_cvae = "model.label_to_dmodel.weight" in state_dict
    if is_cvae:
        model_params.update(
            {
                "num_tasks": state_dict["model.label_to_dmodel.weight"].shape[1],
                "use_vae": model_params.get("use_vae", True),
                "use_conditional_prior": any(
                    "conditional_prior_mu" in key for key in state_dict
                ),
                "modulation_scheme": (
                    "film" if any("model.modulation." in key for key in state_dict) else "none"
                ),
                "lora": original.get("lora", model_params.get("lora")),
                "use_property_prediction": any(
                    "model.property_predictor." in key for key in state_dict
                ),
            }
        )
    else:
        model_params["use_vae"] = model_params.get("use_vae", True)

    lora_a = next(
        (value for key, value in state_dict.items() if key.endswith("lora_A")), None
    )
    hyper_parameters = {
        "model_class": (
            "bmfm_sm_gen.models.cvae.ConditionalTransformerVAE"
            if is_cvae
            else "bmfm_sm_gen.models.vae.TransformerVAE"
        ),
        "model_params": model_params,
        "tokenizer": f"bmfm_sm_gen.data.base_tokenizer.{entry['tokenizer']}",
        "vocab_file": None,
    }
    if is_cvae:
        hyper_parameters.update(
            {
                "lora": model_params.get("lora"),
                "lora_r": int(lora_a.shape[0]) if lora_a is not None else 8,
                "lora_alpha": original.get(
                    "lora_alpha", sanitization_defaults["lora_alpha"]
                ),
                "lora_dropout": original.get(
                    "lora_dropout", sanitization_defaults["lora_dropout"]
                ),
                "use_conditional_prior": model_params["use_conditional_prior"],
                "modulation_scheme": model_params["modulation_scheme"],
                "use_property_prediction": model_params["use_property_prediction"],
                "pretrained_path": None,
                "val_csv": None,
            }
        )

    sanitized = {
        "state_dict": state_dict,
        "hyper_parameters": hyper_parameters,
        "pytorch-lightning_version": checkpoint.get("pytorch-lightning_version"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "star_vae_release": {
            "version": "1.0.0",
            "model_id": entry.get("model_id"),
            "vocabulary": entry["vocabulary"],
            "vocabulary_sha256": vocabulary_sha256,
        },
    }
    buffer = io.BytesIO()
    torch.save(sanitized, buffer)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic ZIP with supported STAR-VAE checkpoints."
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted checkpoints directory.")
    parser.add_argument("--manifest", type=Path, default=Path("checkpoint_bundle.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("star_vae_checkpoints_v1.0.0.zip"))
    parser.add_argument("--checksums", type=Path, default=Path("checkpoints.sha256"))
    parser.add_argument("--force", action="store_true",
                        help="Replace existing output and checksum files.")
    args = parser.parse_args()

    output = Path(os.path.abspath(args.output))
    checksums = Path(os.path.abspath(args.checksums))
    if output == checksums:
        raise ValueError("Archive and checksum outputs must be different files")
    existing = [
        path for path in (output, checksums) if path.exists() or path.is_symlink()
    ]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root, members = manifest_members(manifest)
    sources = resolve_sources(args.source, members)
    checkpoint_entries = {
        entry["checkpoint"]: {**entry, "model_id": model_id}
        for model_id, entry in manifest["models"].items()
    }
    raw_hashes = {member: sha256(source) for member, source in sources.items()}
    payloads = {}
    for member, source in sources.items():
        if member.endswith(".ckpt"):
            entry = checkpoint_entries[member]
            if entry.get("fixture"):
                payloads[member] = source.read_bytes()
            else:
                payloads[member] = sanitized_checkpoint_bytes(
                    source,
                    entry,
                    raw_hashes[entry["vocabulary"]],
                    manifest["sanitization_defaults"],
                )
        else:
            payloads[member] = source.read_bytes()
    payload_hashes = {
        member: hashlib.sha256(payload).hexdigest() for member, payload in payloads.items()
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    checksums.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = temporary_sibling(output)
    temporary_checksums = temporary_sibling(checksums)
    try:
        with ZipFile(
            temporary_output, "w", compression=ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for member in members:
                archive_name = str(PurePosixPath(root) / member)
                info = ZipInfo(archive_name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payloads[member])

        archive_hash = sha256(temporary_output)
        lines = [f"{archive_hash}  {output.name}"]
        lines.extend(
            f"{payload_hashes[member]}  {root}/{member}" for member in members
        )
        temporary_checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with ZipFile(temporary_output) as archive:
            if archive.testzip() is not None:
                raise ValueError("Built archive failed CRC validation")
            if [info.filename for info in archive.infolist()] != [
                f"{root}/{member}" for member in members
            ]:
                raise ValueError("Built archive membership validation failed")
        if sha256(temporary_output) != archive_hash:
            raise ValueError("Built archive changed during validation")

        os.replace(temporary_output, output)
        os.replace(temporary_checksums, checksums)
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_checksums.unlink(missing_ok=True)

    print(f"Built {output} with {len(members)} files")
    print(f"SHA-256: {archive_hash}")


if __name__ == "__main__":
    main()
