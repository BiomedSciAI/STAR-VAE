import numpy as np
import pandas as pd
import torch
from litdata import StreamingDataset
from rdkit import Chem
from rdkit.Chem import QED
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from bmfm_sm_gen.data.base_tokenizer import MolecularTokenizer
from bmfm_sm_gen.utils import sascorer
import selfies as sf

#######################
### DATASET CLASSES ###
#######################


class MolecularDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        max_len: int,
        modality: str,
        tokenizer: MolecularTokenizer,
        label_columns: list[str] | None = None,
        compute_labels: list[str] | None = None,
    ):
        super().__init__()
        usecols = [modality] + (label_columns if label_columns else [])
        self.data = pd.read_csv(csv_file, usecols=usecols)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label_columns = label_columns
        self.compute_labels = compute_labels
        self.modality = modality

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        mol = row[self.modality]
        tokenized = self.tokenizer(
            mol, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        if self.label_columns:
            labels = row[self.label_columns]
            tokenized["labels"] = labels

        if self.compute_labels:
            labels = compute_prop(
                input_str=mol, input_type=self.modality, names=self.compute_labels
            )
            tokenized["labels"] = labels

        return tokenized

    def __len__(self):
        return len(self.data)


class MolecularStreamingDataset(StreamingDataset):
    def __init__(
        self,
        input_dir: str,
        max_len: int,
        modality: str,
        tokenizer: MolecularTokenizer,
        compute_labels: list[str] | None = None,
    ):
        self.input_dir = input_dir
        self.stream_hf = isinstance(input_dir, str) and input_dir.startswith("hf")
        self.max_len = max_len
        self.modality = modality
        self.tokenizer = tokenizer
        self.compute_labels = compute_labels
        super().__init__(input_dir=input_dir)

    def __getitem__(self, idx):
        item_data = super().__getitem__(idx)

        if "selfies" in self.modality:
            mol = item_data["selfies"]
        elif "safe" in self.modality:
            mol = item_data["input"] if self.stream_hf else item_data["safe"]
        elif "smiles" in self.modality:
            mol = item_data["smiles"]
        elif "canonical_selfies" in self.modality:
            mol = item_data["canonical_selfies"]
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

        tokenized = self.tokenizer(
            mol, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        if self.compute_labels:
            labels = compute_prop(
                input_str=mol, input_type=self.modality, names=self.compute_labels
            )
            tokenized["labels"] = labels
        return tokenized


#########################
### UTILITY FUNCTIONS ###
#########################


def build_dataset(
    mode: str,
    max_len: int,
    tokenizer: MolecularTokenizer,
    dataset_args: dict,
    input_dir: str | None = None,
    csv_file: str | None = None,
):
    if mode == "stream":
        return MolecularStreamingDataset(
            input_dir=input_dir,
            max_len=max_len,
            modality=dataset_args.get("modality", "selfies"),
            tokenizer=tokenizer,
            compute_labels=dataset_args.get("compute_labels", None),
        )
    elif mode == "mem":
        return MolecularDataset(
            csv_file=csv_file,
            max_len=max_len,
            modality=dataset_args.get("modality", "selfies"),
            tokenizer=tokenizer,
            label_columns=dataset_args.get("label_columns", None),
            compute_labels=dataset_args.get("compute_labels", None),
        )
    else:
        raise ValueError("mode must be 'stream' or 'mem'.")


def collate_fn(
    batch: list[dict],
    tokenizer: MolecularTokenizer,
    padding: bool = "longest",
    max_length: int | None = None,
) -> dict:
    if padding not in ("longest", "max_length"):
        raise ValueError("padding must be one of: 'longest', 'max_length'")
    if padding == "max_length" and max_length is None:
        raise ValueError("max_length must be set when padding='max_length'")

    def to_tensor_list(key):
        return [
            (
                torch.tensor(item[key], dtype=torch.int64)
                if not isinstance(item[key], torch.Tensor)
                else item[key]
            )
            for item in batch
        ]

    input_ids_list = to_tensor_list("input_ids")
    attention_mask_list = to_tensor_list("attention_mask")
    token_type_ids_list = to_tensor_list("token_type_ids")
    label_list = [
        torch.tensor(item["labels"], dtype=torch.float32)
        for item in batch
        if "labels" in item
    ]

    # Pad to the longest sequence
    input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0
    )
    token_type_ids = pad_sequence(
        token_type_ids_list, batch_first=True, padding_value=0
    )
    labels = (
        pad_sequence(label_list, batch_first=True, padding_value=0)
        if label_list
        else None
    )
    # If padding to max_length, truncate after padding
    if padding == "max_length":
        input_ids = input_ids[:, :max_length]
        attention_mask = attention_mask[:, :max_length]
        token_type_ids = token_type_ids[:, :max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }


def compute_prop(input_str: str, input_type: str = "smiles", names=None) -> dict:
    input_type = input_type.lower()
    if input_type == "selfies":
        try:
            smiles = sf.decoder(input_str)
        except Exception as e:
            raise ValueError(f"Invalid SELFIES: {e}")
    elif input_type in {"smiles", "safe"}:
        smiles = input_str
    else:
        raise ValueError("input_type must be one of: 'smiles', 'selfies', 'safe'")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    labels = []
    if "sa_score" in names:
        sa_score = sascorer.calculateScore(mol)
        labels.append(sa_score)
    if "qed" in names:
        qed_score = QED.qed(mol)
        labels.append(qed_score)
    return labels
