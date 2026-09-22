import importlib
import inspect
import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from rdkit import Chem
from rdkit.Chem import Draw


def import_class(class_path):
    if type(class_path) is str:
        split_path = class_path.rsplit(".", 1)
        module_name, class_name = split_path[0], split_path[1]
        return getattr(importlib.import_module(module_name), class_name)
    else:
        return class_path


def subset_dict(cls, dict_args):
    sig = inspect.signature(cls)
    allowed = set(sig.parameters)
    pass_through = {k: v for k, v in dict_args.items() if k in allowed}
    unknown = set(dict_args) - allowed
    logging.warning(
        f"{cls} does not accept argument(s): {', '.join(unknown)}. Ignoring them. Letting through: {pass_through}"
    )
    return pass_through


def PIL_to_tensor(img):
    img = img.convert("RGB")
    img = np.array(img)
    img = torch.tensor(img).permute(2, 0, 1).float()  # Convert to (C, H, W)
    return img / 255.0


def draw_labeled_molecule_grid(labeled_molecules, mols_per_row=4, max_mols=16):
    labeled_molecules = labeled_molecules[:max_mols]

    mols = []
    legends = []

    for smi, intended_label, computed_score in labeled_molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            caption = f"{intended_label:.2f}, {computed_score:.2f}"
            legends.append(caption)

    if not mols:
        return None

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(200, 200),
        legends=legends,
        useSVG=False,
    )
    return img


class LossPlotter(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = (
            outputs["loss"].detach().cpu().item() if isinstance(outputs, dict) else None
        )
        if loss is not None:
            self.train_loss.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            self.val_loss.append(val_loss.cpu().item())

    def on_train_end(self, trainer, pl_module):
        plt.figure()
        plt.plot(self.train_loss, label="Train loss (per batch)")
        plt.plot(
            np.linspace(0, len(self.train_loss), len(self.val_loss)),
            self.val_loss,
            label="Val loss (per epoch)",
        )
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.show()
