import logging
import os
from functools import partial

import pytorch_lightning as pl
from litdata import StreamingDataLoader
from torch.utils.data import DataLoader

from bmfm_sm_gen.data.ds_mol import build_dataset, collate_fn
from bmfm_sm_gen.utils.trainer_utils import import_class


class MolecularDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer: str,
        vocab_file: str = None,
        max_len: int = 74,
        batch_size: int = 64,
        train_mode: str = "stream",
        train_csv: str = None,
        val_csv: str = None,
        allow_same_split: bool = False,
        dataset_args: dict = {},
    ):
        """
        Data module for generative models.

        Args:
            - data_dir --> Point to either a directory containing litdata shards or a HF URI
            - tokenizer --> String path to tokenizer class (e.g. bmfm_sm_gen.data.base_tokenizer.SELFIESTokenizer)
            - max_len --> Maximum length of a token sequence (Should be compatible with model)
            - batch_size --> Batch size for the data_loader
            - train_mode --> Either "mem" or "stream"; If mem, then train_csv and val_csv must be specified (Local files); If stream, then provide data_dir
            - train_csv / val_csv --> Paths to CSV files (Only required if train_mode is "mem")
            - allow_same_split --> Explicitly permit train_csv and val_csv to resolve to the same file (default: False)
            - dataset_args -->
                - modality --> String specified the modality (['selfies', 'safes', 'smiles']); Should match CSV column name / shard dictionary key
                - padding --> True / "max_length" / "longest" (True and longest are equivalent, will pad to the length of the longest item in the batch. "max_length" pads to max_len)

        """
        super().__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.allow_same_split = allow_same_split
        self.mode = train_mode
        self.dataset_args = dataset_args

        if isinstance(tokenizer, str):
            tokenizer = import_class(tokenizer)
        self.tokenizer = tokenizer(vocab=vocab_file)

    def prepare_data(self):
        logging.info("prepare_data() - done.")

    def setup(self, stage=None):
        if (
            (stage == "fit" or getattr(stage, "value", None) == "fit")
            and self.train_mode == "mem"
            and self.train_csv
            and self.val_csv
            and not self.allow_same_split
            and os.path.realpath(os.path.abspath(self.train_csv))
            == os.path.realpath(os.path.abspath(self.val_csv))
        ):
            raise ValueError(
                "train_csv and val_csv resolve to the same file. Provide distinct "
                "splits or set allow_same_split=True explicitly."
            )

        if isinstance(self.data_dir, str) and self.data_dir.startswith("hf"):
            input_dirs = {
                "train": self.data_dir.rstrip("/") + "/train-*.parquet",
                "val": self.data_dir.rstrip("/") + "/validation-*.parquet",
            }
        elif self.data_dir:
            input_dirs = {
                "train": os.path.join(self.data_dir, "train"),
                "val": os.path.join(self.data_dir, "val"),
            }
        else:
            input_dirs = {"train": None, "val": None}

        self.train_dataset = build_dataset(
            mode=self.train_mode,
            max_len=self.max_len,
            tokenizer=self.tokenizer,
            dataset_args=self.dataset_args,
            input_dir=input_dirs["train"],
            csv_file=self.train_csv,
        )
        self.val_dataset = build_dataset(
            mode=self.train_mode,
            max_len=self.max_len,
            tokenizer=self.tokenizer,
            dataset_args=self.dataset_args,
            input_dir=input_dirs["val"],
            csv_file=self.val_csv,
        )
        logging.info("setup() - done.")

    def train_dataloader(self):
        LoaderClass = DataLoader if self.train_mode == "mem" else StreamingDataLoader
        return LoaderClass(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=partial(
                collate_fn,
                padding=self.dataset_args.get("padding", "longest"),
                tokenizer=self.tokenizer,
                max_length=self.max_len,
            ),
        )

    def val_dataloader(self):
        LoaderClass = DataLoader if self.train_mode == "mem" else StreamingDataLoader
        return LoaderClass(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=partial(
                collate_fn,
                padding=self.dataset_args.get("padding", "longest"),
                tokenizer=self.tokenizer,
                max_length=self.max_len,
            ),
        )

    def predict_dataloader(self):
        LoaderClass = DataLoader if self.train_mode == "mem" else StreamingDataLoader
        return LoaderClass(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=partial(
                collate_fn,
                padding=self.dataset_args.get("padding", "longest"),
                tokenizer=self.tokenizer,
                max_length=self.max_len,
            ),
        )
