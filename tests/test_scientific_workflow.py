import importlib.util
import logging
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


def _load_exp19_module():
    path = REPO_ROOT / "bmfm_sm_gen/scripts/exp19_generate_ablation.py"
    spec = importlib.util.spec_from_file_location("exp19_generate_ablation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_moses_selfies_config_uses_released_checkpoint_vocab():
    config_path = REPO_ROOT / "configs/model_vae_train_moses.yaml"
    config = yaml.safe_load(config_path.read_text())
    vocab_setting = config["data"]["init_args"]["vocab_file"]
    assert vocab_setting == "${oc.env:VOCAB_FILE,checkpoints/vae_pubchem80m/vocab.json}"

    assert "vae_pubchem80m/vocab.json" in vocab_setting


def test_exp19_generation_preserves_attempts_and_validity(caplog):
    exp19 = _load_exp19_module()

    class StubModel:
        def generate_molecules_with_target_label(self, num_samples, **kwargs):
            assert num_samples == 3
            return ["CC", None, "O"]

    with caplog.at_level(logging.INFO):
        result = exp19.generate_for_target(
            StubModel(), target=0.5, n_samples=3, seed=42, device="cpu"
        )

    assert result.to_dict("list") == {
        "attempt_id": [0, 1, 2],
        "smiles": ["CC", None, "O"],
        "valid": [True, False, True],
        "target": [0.5, 0.5, 0.5],
        "seed": [42, 42, 42],
    }
    assert "attempted=3 valid=2 invalid=1" in caplog.text


def test_exp19_failed_batch_is_recorded_as_invalid():
    exp19 = _load_exp19_module()

    class FailingModel:
        def generate_molecules_with_target_label(self, **kwargs):
            raise RuntimeError("sampling failed")

    result = exp19.generate_for_target(
        FailingModel(), target=0.5, n_samples=2, seed=42, device="cpu"
    )
    assert result["attempt_id"].tolist() == [0, 1]
    assert result["smiles"].isna().all()
    assert not result["valid"].any()


def test_datamodule_rejects_resolved_identical_split_paths(tmp_path, vocab_file):
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule

    csv_path = tmp_path / "molecules.csv"
    csv_path.write_text("selfies\n[C]\n")
    equivalent_path = tmp_path / "split_alias.csv"
    equivalent_path.symlink_to(csv_path)
    dm = MolecularDataModule(
        data_dir="",
        train_mode="mem",
        train_csv=str(csv_path),
        val_csv=str(equivalent_path),
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
    )

    with pytest.raises(ValueError, match="resolve to the same file"):
        dm.setup("fit")


def test_datamodule_allows_explicit_same_split_opt_in(tmp_path, vocab_file):
    from bmfm_sm_gen.data.dm_mol import MolecularDataModule

    csv_path = tmp_path / "molecules.csv"
    csv_path.write_text("selfies\n[C]\n")
    dm = MolecularDataModule(
        data_dir="",
        train_mode="mem",
        train_csv=str(csv_path),
        val_csv=str(csv_path),
        allow_same_split=True,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer",
        vocab_file=vocab_file,
    )

    dm.setup("fit")
    assert len(dm.train_dataset) == len(dm.val_dataset) == 1


def test_cvae_generation_validation_is_opt_in(tiny_cvae, monkeypatch):
    from bmfm_sm_gen.trainers import lm_cvae

    monkeypatch.setattr(
        lm_cvae,
        "run_conditional_generation_evaluation_compute_prop",
        lambda **kwargs: pytest.fail("generation validation ran unexpectedly"),
    )
    tiny_cvae.on_validation_epoch_end()
