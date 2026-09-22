import logging
import os
import sys
from glob import glob

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

# NOTE: 'from clearml import Task' is intentionally NOT at module level.
# ClearML 1.18.0 patches jsonargparse.adapt_typehints at import time.
# That patch breaks LightningCLI(subclass_mode_model=True) for any model
# class with complex Union-typed init args (e.g. CVAELightningModule).
# The import is deferred to inside main() after the enabled-check so the
# patch is never installed when ClearML is disabled.


def find_latest_checkpoint(directory: str):
    ckpt_files = glob(os.path.join(directory, "*.ckpt"))
    if not ckpt_files:
        return None
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    return ckpt_files[0]


class CLI(LightningCLI):
    def before_instantiate_classes(self):
        logging.info(f"Running {self.subcommand} using configuration {self.config}")
        config = (
            self.config[self.config["subcommand"]]
            if "subcommand" in self.config
            else self.config
        )
        ckpt_path = os.path.join(
            config.get("trainer.default_root_dir"), "checkpoints", "last.ckpt"
        )
        ckpt_found = os.path.isfile(ckpt_path)
        if ckpt_found:
            logging.info(f"Using checkpoint {ckpt_path}")
            config["ckpt_path"] = ckpt_path
        else:
            logging.info(
                f"No checkpoint found. Training from scratch. Path: {ckpt_path}"
            )


def main():
    # Initialise ClearML BEFORE constructing LightningCLI.  If Task.init() is
    # called inside a CLI hook (after CLI construction begins), ClearML patches
    # jsonargparse's adapt_typehints mid-parse, corrupting Union[None, LightningModule]
    # type resolution and crashing with "Does not validate against any Union subtype".
    clearml_enabled = os.getenv("CLEARML_ENABLE", "False").lower() == "true"
    is_rank_zero = (
        os.environ.get("LOCAL_RANK", "0") == "0"
        and os.environ.get("NODE_RANK", "0") == "0"
    )
    if clearml_enabled:
        # Lazy import: MUST be inside this block so ClearML's module-level
        # jsonargparse monkey-patch (ClearML 1.18.0) is never installed when
        # ClearML is disabled. Top-level import would patch jsonargparse for
        # every run and break LightningCLI(subclass_mode_model=True).
        try:
            from clearml import Task  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            if exc.name != "clearml":
                raise
            raise RuntimeError(
                "CLEARML_ENABLE=true requires the ClearML extra. "
                "Install it with `pip install 'star-vae[clearml]'` or "
                "`uv sync --extra clearml`."
            ) from exc

    if is_rank_zero and clearml_enabled:
        # Use env vars for project/task, --clearml.* CLI args must not reach
        # jsonargparse after the patch is installed.
        project_name = os.getenv("CLEARML_PROJECT", "bmfm-sm")
        task_name    = os.getenv("CLEARML_TASK_NAME", "default_experiment")
        # Strip --clearml.* from argv in case they were passed the old way
        filtered_argv = []
        i = 0
        while i < len(sys.argv):
            if sys.argv[i] in ("--clearml.project_name", "--clearml.task_name") \
                    and i + 1 < len(sys.argv):
                i += 2
            else:
                filtered_argv.append(sys.argv[i]); i += 1
        sys.argv = filtered_argv
        Task.init(
            project_name=project_name,
            task_name=task_name,
            reuse_last_task_id=True,
            continue_last_task=True,
        )

    CLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    main()
