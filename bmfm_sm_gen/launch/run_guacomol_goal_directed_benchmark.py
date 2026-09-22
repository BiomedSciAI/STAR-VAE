from __future__ import annotations

import argparse
import math
import time

import random
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from joblib import Parallel, delayed
import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as _np, scipy as _sp
import selfies as sf
import datetime, os, json
from tqdm import tqdm

from bmfm_sm_gen.utils.scaffold_utils import (
    compute_scaffolds,
    rank_scaffolds_label,
    top_diverse_molecules_within_top_scaffolds,
)

if not hasattr(_sp, "histogram"):
    _sp.histogram = _np.histogram

import numpy as np
import torch
import torch.nn.functional as F
import dataclasses


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
import cma
from bmfm_sm_gen.trainers.lm_vae import VAELightningModule
from bmfm_sm_gen.data.base_tokenizer import SELFIESTokenizer
from guacamol.utils.chemistry import canonicalize


logger = logging.getLogger("vae_goal_directed")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s - VAE-GD - %(levelname)s - %(message)s")
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def calc_k_per_seed(population: int, rounds: int, budget: int) -> int:
    if population <= 0 or rounds <= 0:
        return 1
    k = budget // max(1, (population * rounds))
    return max(1, int(k))


def planned_total(population: int, k_per_seed: int, rounds: int) -> int:
    return int(population) * int(k_per_seed) * int(rounds)


def log_round_stats(logger, round_idx, gen_df, label_col="label_lb"):
    n_valid = len(gen_df)
    n_unique = gen_df["std_smiles"].nunique()
    n_scaf = gen_df["Murcko Scaffold"].nunique(dropna=True)
    mean_s = float(gen_df[label_col].mean())
    max_s = float(gen_df[label_col].max())
    logger.info(
        f"[scaf] R{round_idx+1:02d} stats | valid={n_valid:,} | unique={n_unique:,} | "
        f"scaffolds={n_scaf:,} | mean(lb)={mean_s:.3f} | max(lb)={max_s:.3f}"
    )


def score_smiles(sf: Any, smi: str) -> float:
    try:
        return float(sf(smi))
    except TypeError:
        pass
    except Exception:
        raise

    for meth in ("score", "score_smiles", "score_single"):
        if hasattr(sf, meth) and callable(getattr(sf, meth)):
            return float(getattr(sf, meth)(smi))

    for attr in ("objective", "raw_objective", "wrapped_objective"):
        obj = getattr(sf, attr, None)
        if obj is not None:
            try:
                return float(obj(smi))
            except TypeError:
                pass
            for meth in ("score", "score_smiles", "score_single"):
                if hasattr(obj, meth) and callable(getattr(obj, meth)):
                    return float(getattr(obj, meth)(smi))

    try:
        out = sf([smi])
        if isinstance(out, (list, tuple, np.ndarray)) and len(out) > 0:
            return float(out[0])
    except Exception:
        pass

    raise TypeError(
        "Incompatible scoring function wrapper: not callable and no usable .score/.score_* found."
    )


def score_list(sf: Any, smiles: List[str]) -> List[float]:
    for meth in ("score_list", "score_many"):
        if hasattr(sf, meth) and callable(getattr(sf, meth)):
            return [float(x) for x in getattr(sf, meth)(smiles)]

    try:
        out = sf(smiles)
        if isinstance(out, (list, tuple, np.ndarray)) and len(out) == len(smiles):
            return [float(x) for x in out]
    except TypeError:
        pass

    for attr in ("objective", "raw_objective", "wrapped_objective"):
        obj = getattr(sf, attr, None)
        if obj is not None:
            try:
                out = obj(smiles)
                if isinstance(out, (list, tuple, np.ndarray)) and len(out) == len(
                    smiles
                ):
                    return [float(x) for x in out]
            except TypeError:
                pass
            for meth in ("score_list", "score_many"):
                if hasattr(obj, meth) and callable(getattr(obj, meth)):
                    return [float(x) for x in getattr(obj, meth)(smiles)]

    return [score_smiles(sf, s) for s in smiles]


def smiles_is_valid(smi: Optional[str]) -> bool:
    if not smi:
        return False
    try:
        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False


def fp_from_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def diversity_filter(smiles: List[str], min_tanimoto_dist: float = 0.2) -> List[str]:
    kept, fps = [], []
    for smi in smiles:
        if not smiles_is_valid(smi):
            continue
        fp = fp_from_smiles(smi)
        if fp is None:
            continue
        if not fps:
            kept.append(smi)
            fps.append(fp)
            continue
        ok = True
        for fp_prev in fps:
            sim = DataStructs.TanimotoSimilarity(fp, fp_prev)
            if (1.0 - sim) < min_tanimoto_dist:
                ok = False
                break
        if ok:
            kept.append(smi)
            fps.append(fp)
    return kept


class VAEModelWrapper:
    def __init__(
        self, model, tokenizer, latent_dim: int, device: Optional[torch.device] = None
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.latent_dim = latent_dim
        self.device = device or next(self.model.parameters()).device

    @torch.no_grad()
    def decode_latents(
        self,
        z: torch.Tensor,
        max_len: Optional[int] = None,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        sampling_scheme: str = "around_mu",
        std_dev: float = 1.0,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> list[str]:
        self.model.eval()
        device = self.device
        max_len = max_len or getattr(self.model, "max_len", 72)

        if sampling_scheme == "around_mu" and (mu is not None and logvar is not None):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * std_dev
            z = mu + eps * std
        elif sampling_scheme == "prior":
            z = torch.randn_like(z) * std_dev

        z = z.to(device)
        B = z.size(0)

        z_hidden = self.model.latent_to_hidden(z)
        memory = z_hidden.unsqueeze(1).repeat(1, max_len, 1).transpose(0, 1)

        generated = torch.full(
            (B, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device
        )

        for _ in range(max_len - 1):
            tgt = generated
            tgt_emb = self.model.decoder.embedding(tgt) * math.sqrt(
                self.model.decoder.d_model
            )
            tgt_emb = self.model.decoder.pos_decoder(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1)

            tgt_mask = self.model.decoder.generate_square_subsequent_mask(
                tgt_emb.size(0)
            ).to(device)
            tgt_kpm = tgt == self.tokenizer.pad_token_id

            out = self.model.decoder.transformer_decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_kpm,
            ).transpose(0, 1)

            logits = self.model.decoder.output_layer(out[:, -1, :]) / temperature

            if top_k is not None and top_k > 1:
                values, indices = torch.topk(logits, k=top_k, dim=-1)
                probs = F.softmax(values, dim=-1)
                # map multinomial back to original vocab indices
                choice_in_topk = torch.multinomial(probs, num_samples=1)
                next_tok = indices.gather(-1, choice_in_topk)
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tok], dim=1)
            if (next_tok == self.tokenizer.eos_token_id).all():
                break

        seqs = generated[:, 1:]
        smiles = []
        for seq in seqs:
            tokens = []
            for idx in seq.tolist():
                if idx in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
                    break
                tokens.append(self.tokenizer.convert_ids_to_tokens(idx))
            selfies_str = "".join(tokens)
            try:
                smi = sf.decoder(selfies_str)
                smi = smi if smiles_is_valid(smi) else None
            except Exception:
                smi = None
            smiles.append(smi)
        return smiles

    def encode_smiles(self, smiles_list):
        self.model.eval()
        latents = []
        with torch.no_grad():
            for smi in smiles_list:
                try:
                    selfies_str = sf.encoder(smi)
                    tokens = self.tokenizer.encode(selfies_str)
                    input_tensor = torch.tensor(
                        tokens, dtype=torch.long, device=self.device
                    ).unsqueeze(0)
                    mu, _ = self.model.encode(input_tensor)
                    latents.append(mu.squeeze(0))
                except Exception:
                    latents.append(torch.randn(self.latent_dim, device=self.device))
        return torch.stack(latents, dim=0)


@dataclass
class StrategyConfig:
    population: int = 512
    survivors: int = 64
    rounds: int = 30
    sigma_start: float = 0.12
    sigma_end: float = 0.02
    diversity_dist: float = 0.2
    max_decode_len: Optional[int] = None
    seed: int = 0
    log_every: int = 2

    # scaffold-frontier specific
    k_per_seed: int = 64
    top_scaffolds: int = 20
    per_scaffold: int = 5

    # GA-specific
    offspring_size: int = 200  # number of individuals in mating pool
    mutation_rate: float = 0.5  # mutation probability per child
    mutation_sigma: float = 0.1  # mutation strength in latent space
    patience: int = 5  # stop if n
    smiles_file: str = None


class LatentStrategy:
    def __init__(self, model: VAEModelWrapper, cfg: StrategyConfig):
        self.model = model
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    def run(self, scoring_function: Any, budget_mols: int) -> List[str]:
        raise NotImplementedError


class RandomSamplingStrategy(LatentStrategy):
    def run(self, scoring_function, budget_mols: int) -> List[str]:
        logger.info("[random] sampling %d latents", self.cfg.population)
        z = torch.randn(
            self.cfg.population, self.model.latent_dim, device=self.model.device
        )
        smiles = self.model.decode_latents(z, max_len=self.cfg.max_decode_len)
        valid = [s for s in smiles if s and smiles_is_valid(s)]
        scores = score_list(scoring_function, valid)
        cand = list(zip(scores, valid))
        cand.sort(key=lambda x: x[0], reverse=True)
        ranked = [s for _, s in cand]
        ranked = diversity_filter(ranked, self.cfg.diversity_dist)
        logger.info(
            "[random] valid=%d best=%.3f",
            len(valid),
            cand[0][0] if cand else float("nan"),
        )
        return ranked[:budget_mols]


class HillClimbStrategy(LatentStrategy):
    def run(self, scoring_function, budget_mols: int) -> List[str]:
        pop, keep, rounds = self.cfg.population, self.cfg.survivors, self.cfg.rounds
        sigma = self.cfg.sigma_start
        sigma_decay = (self.cfg.sigma_end / max(1e-12, self.cfg.sigma_start)) ** (
            1.0 / max(1, rounds)
        )
        z = torch.randn(pop, self.model.latent_dim, device=self.model.device)
        hall: List[Tuple[float, str]] = []

        for r in range(rounds):
            smiles = self.model.decode_latents(z, max_len=self.cfg.max_decode_len)
            valid = [(i, s) for i, s in enumerate(smiles) if s and smiles_is_valid(s)]
            if not valid:
                z = torch.randn(pop, self.model.latent_dim, device=self.model.device)
                continue
            idxs, val_smiles = zip(*valid)
            scores = score_list(scoring_function, list(val_smiles))
            scored = sorted(
                [(sc, s, i) for (i, s), sc in zip(valid, scores)],
                key=lambda x: x[0],
                reverse=True,
            )
            hall.extend([(sc, s) for sc, s, _ in scored[:keep]])

            parent_idx = [i for _, _, i in scored[:keep]]
            parents = z[list(parent_idx)]
            mutated = parents + torch.randn_like(parents) * sigma
            if mutated.size(0) < pop:
                randfill = torch.randn(
                    pop - mutated.size(0),
                    self.model.latent_dim,
                    device=self.model.device,
                )
                z = torch.cat([mutated, randfill], dim=0)
            else:
                z = mutated[:pop]
            sigma = max(self.cfg.sigma_end, sigma * sigma_decay)

            if (r + 1) % self.cfg.log_every == 0:
                logger.info(
                    "[hill] round %d/%d: valid=%d best=%.3f sigma=%.3f",
                    r + 1,
                    rounds,
                    len(valid),
                    scored[0][0],
                    sigma,
                )

        hall.sort(key=lambda x: x[0], reverse=True)
        uniq = diversity_filter([s for _, s in hall], self.cfg.diversity_dist)
        return uniq[:budget_mols]


class EvolutionaryStrategy(LatentStrategy):
    """(μ+λ) evolution in latent space."""

    def run(self, scoring_function, budget_mols: int) -> List[str]:
        mu = self.cfg.survivors
        rounds = self.cfg.rounds
        sigma = self.cfg.sigma_start
        sigma_decay = (self.cfg.sigma_end / max(1e-12, self.cfg.sigma_start)) ** (
            1.0 / max(1, rounds)
        )

        elites = torch.randn(mu, self.model.latent_dim, device=self.model.device)
        hall: List[Tuple[float, str]] = []

        for r in range(rounds):
            children = elites + sigma * torch.randn_like(elites)
            batch = torch.cat([elites, children], dim=0)  # (2μ)
            smiles = self.model.decode_latents(batch, max_len=self.cfg.max_decode_len)
            valid = [(i, s) for i, s in enumerate(smiles) if s and smiles_is_valid(s)]
            if not valid:
                elites = torch.randn(
                    mu, self.model.latent_dim, device=self.model.device
                )
                continue
            idxs, val_smiles = zip(*valid)
            scores = score_list(scoring_function, list(val_smiles))
            scored = sorted(
                [(sc, s, i) for (i, s), sc in zip(valid, scores)],
                key=lambda x: x[0],
                reverse=True,
            )

            best = scored[:mu]
            elites = batch[[i for _, _, i in best]]
            hall.extend([(sc, s) for sc, s, _ in best])
            sigma = max(self.cfg.sigma_end, sigma * sigma_decay)

            if (r + 1) % self.cfg.log_every == 0:
                logger.info(
                    "[evo] round %d/%d: valid=%d best=%.3f sigma=%.3f",
                    r + 1,
                    rounds,
                    len(valid),
                    scored[0][0],
                    sigma,
                )

        hall.sort(key=lambda x: x[0], reverse=True)
        uniq = diversity_filter([s for _, s in hall], self.cfg.diversity_dist)
        return uniq[:budget_mols]


class SimulatedAnnealingStrategy(LatentStrategy):
    def run(self, scoring_function, budget_mols: int) -> List[str]:
        rounds, pop = self.cfg.rounds, self.cfg.population
        T0, Tend = 1.0, 0.05
        T_decay = (Tend / T0) ** (1.0 / max(1, rounds))

        z = torch.randn(pop, self.model.latent_dim, device=self.model.device)
        smiles = self.model.decode_latents(z, max_len=self.cfg.max_decode_len)
        valid = [s for s in smiles if s and smiles_is_valid(s)]
        vals = np.array(score_list(scoring_function, valid)) if valid else np.zeros(0)

        hall: List[Tuple[float, str]] = []
        T = T0
        for r in range(rounds):
            z_prop = z + self.cfg.sigma_start * torch.randn_like(z) * max(T, 1e-3)
            smiles_p = self.model.decode_latents(
                z_prop, max_len=self.cfg.max_decode_len
            )
            valid_p = [s for s in smiles_p if s and smiles_is_valid(s)]
            vals_p = (
                np.array(score_list(scoring_function, valid_p))
                if valid_p
                else np.zeros(0)
            )

            n = (
                min(len(valid), len(vals), len(valid_p), len(vals_p))
                if (len(valid) and len(valid_p))
                else 0
            )
            if n > 0:
                prob = np.exp((vals_p[:n] - vals[:n]) / max(T, 1e-8))
                accept = (vals_p[:n] > vals[:n]) | (np.random.rand(n) < prob)
                # use only first n entries for alignment
                z[:n][torch.as_tensor(accept, dtype=torch.bool)] = z_prop[:n][
                    torch.as_tensor(accept, dtype=torch.bool)
                ]
                vals[:n][accept] = vals_p[:n][accept]
                valid[:n] = [valid_p[i] if accept[i] else valid[i] for i in range(n)]

            T *= T_decay
            if (r + 1) % self.cfg.log_every == 0:
                best = float(vals.max()) if vals.size else float("nan")
                logger.info(
                    "[sa] round %d/%d: best=%.3f T=%.3f", r + 1, rounds, best, T
                )

        hall = list(zip(vals.tolist(), valid))
        hall.sort(key=lambda x: x[0], reverse=True)
        uniq = diversity_filter([s for _, s in hall], self.cfg.diversity_dist)
        return uniq[:budget_mols]


class CMAESStrategy(LatentStrategy):
    """CMA-ES in latent space."""

    def run(self, scoring_function, budget_mols: int) -> List[str]:
        dim = self.model.latent_dim
        sigma0 = self.cfg.sigma_start
        opts = cma.CMAOptions()
        opts.set("popsize", self.cfg.population)
        opts.set("maxiter", self.cfg.rounds)
        opts.set("verb_disp", 0)
        es = cma.CMAEvolutionStrategy(np.zeros(dim), sigma0, opts)

        hall: List[Tuple[float, str]] = []
        r = 0
        while not es.stop():
            X = np.array(es.ask())
            z = torch.tensor(X, dtype=torch.float32, device=self.model.device)
            smiles = self.model.decode_latents(z, max_len=self.cfg.max_decode_len)
            valid = [s for s in smiles if s and smiles_is_valid(s)]
            if not valid:
                # penalize this step
                es.tell(X, [1e2] * len(X))
                r += 1
                continue
            scores = score_list(scoring_function, valid)
            score_map = {s: sc for s, sc in zip(valid, scores)}
            losses = [-score_map.get(s, 0.0) for s in smiles]
            es.tell(X, losses)
            hall.extend([(score_map.get(s, 0.0), s) for s in valid])
            r += 1
            if (r % self.cfg.log_every) == 0:
                logger.info(
                    "[cma] round %d: valid=%d best=%.3f",
                    r,
                    len(valid),
                    max(scores) if scores else float("nan"),
                )

        hall.sort(key=lambda x: x[0], reverse=True)
        uniq = diversity_filter([s for _, s in hall], self.cfg.diversity_dist)
        return uniq[:budget_mols]


class LatentGAStrategy(LatentStrategy):
    """adapted from Jensen88"""

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        smiles_file = getattr(cfg, "smiles_file", None)
        logger.info(f"About to load {smiles_file} dataset smiles.")
        self.dataset_smiles = self.load_smiles_from_file(smiles_file)
        logger.info(f"Finished loading {len(self.dataset_smiles)} dataset smiles.")

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            smiles_list = [s.strip() for s in f]  # Eagerly read
        jobs = [delayed(canonicalize)(s) for s in smiles_list]
        return Parallel(n_jobs=-1, backend="threading")(jobs)

    def _top_k_from_dataset(self, scoring_function) -> List[str]:
        if not self.dataset_smiles:
            return []
        scores = score_list(scoring_function, self.dataset_smiles)
        scored = sorted(
            zip(scores, self.dataset_smiles), key=lambda t: t[0], reverse=True
        )
        top = [s for _, s in scored[: self.cfg.population]]
        logger.info(
            f"[ga-init] selected {len(top)} / {len(self.dataset_smiles)} top molecules "
            f"(max={scored[0][0]:.3f}, min={scored[self.cfg.population-1][0]:.3f})"
        )
        return top

    def _init_population(
        self, scoring_function, seed_smiles: List[str] | None
    ) -> List[str]:
        if seed_smiles:
            return seed_smiles[: self.cfg.population]
        top_dataset = self._top_k_from_dataset(scoring_function)
        if top_dataset:
            return top_dataset
        Z = torch.randn(
            self.cfg.population, self.model.latent_dim, device=self.model.device
        )
        S = self.model.decode_latents(Z, max_len=self.cfg.max_decode_len)
        return [s for s in S if s and smiles_is_valid(s)][: self.cfg.population]

    def _encode_batch(self, smiles: List[str]) -> torch.Tensor:
        latents = []
        for s in smiles:
            if not s:
                continue
            try:
                z = self.model.encode_smiles([s])  # always pass as list!
                if isinstance(z, torch.Tensor):
                    # expect shape [1, latent_dim]
                    z = z.squeeze(0)
                elif isinstance(z, list) and isinstance(z[0], torch.Tensor):
                    z = z[0].squeeze(0)
                else:
                    z = torch.tensor(z, dtype=torch.float32, device=self.model.device)
            except Exception:
                z = torch.randn(self.model.latent_dim, device=self.model.device)
            latents.append(z)
        return torch.stack(latents, dim=0).to(self.model.device)

    def _crossover_pair(self, zA: torch.Tensor, zB: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(1, device=zA.device).item()
        return alpha * zA + (1.0 - alpha) * zB

    def _mutate(self, z: torch.Tensor) -> torch.Tensor:
        if random.random() > self.cfg.mutation_rate:
            return z
        return z + self.cfg.mutation_sigma * torch.randn_like(z)

    def _roulette_sample(
        self, items: List[str], scores: List[float], k: int
    ) -> List[str]:
        x = np.array(scores, dtype=np.float64)
        x = x - x.min() + 1e-12
        p = x / x.sum() if x.sum() > 0 else np.full_like(x, 1.0 / len(x))
        idx = np.random.choice(len(items), size=k, replace=True, p=p)
        return [items[i] for i in idx]

    def run(
        self, scoring_function, budget_mols: int, seed_smiles: List[str] | None = None
    ) -> List[str]:
        pop = self._init_population(scoring_function, seed_smiles)
        if not pop:
            return []

        hall: List[Tuple[float, str]] = []
        patience, best = 0, -1e9

        for r in range(self.cfg.rounds):
            scores = score_list(scoring_function, pop)
            hall.extend(zip(scores, pop))

            uniq_pop = diversity_filter(pop, self.cfg.diversity_dist)
            uniq_scores_map = {s: sc for s, sc in zip(pop, scores)}
            uniq_scores = [uniq_scores_map.get(s, 0.0) for s in uniq_pop]
            top = sorted(zip(uniq_scores, uniq_pop), key=lambda t: t[0], reverse=True)[
                : self.cfg.population
            ]
            pop = [s for _, s in top]
            scores = [sc for sc, _ in top]

            cur_best = max(scores) if scores else -1e9
            if cur_best <= best + 1e-9:
                patience += 1
                if patience >= self.cfg.patience:
                    logger.info(f"[ga] early stop at generation {r} (no improvement)")
                    break
            else:
                best, patience = cur_best, 0

            # --- mating and reproduction ---
            mating_pool = self._roulette_sample(pop, scores, self.cfg.offspring_size)
            Z_par = self._encode_batch(mating_pool)
            parents_latent = [Z_par[i] for i in range(len(mating_pool))]

            children_latent = []
            for _ in range(self.cfg.population):
                i, j = np.random.randint(0, len(parents_latent)), np.random.randint(
                    0, len(parents_latent)
                )
                zc = self._crossover_pair(parents_latent[i], parents_latent[j])
                zc = self._mutate(zc)
                children_latent.append(zc.unsqueeze(0))
            Z_child = torch.cat(children_latent, dim=0)

            smiles_child = self.model.decode_latents(
                Z_child, max_len=self.cfg.max_decode_len
            )
            smiles_child = [s for s in smiles_child if s and smiles_is_valid(s)]
            if not smiles_child:
                continue

            scores_child = score_list(scoring_function, smiles_child)
            hall.extend(zip(scores_child, smiles_child))

            merged = pop + smiles_child
            merged_scores = scores + scores_child
            pick = sorted(zip(merged_scores, merged), key=lambda t: t[0], reverse=True)[
                : self.cfg.population
            ]
            pop = [s for _, s in pick]

            if (r + 1) % self.cfg.log_every == 0:
                logger.info(
                    f"[ga] gen={r+1:03d} best={max(scores):.3f} mean={np.mean(scores):.3f}"
                )

        hall.sort(key=lambda x: x[0], reverse=True)
        uniq = diversity_filter([s for _, s in hall], self.cfg.diversity_dist)
        return uniq[:budget_mols]


class ScaffoldFrontierStrategy:

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.population = getattr(cfg, "population", 512)
        self.n_rounds = getattr(cfg, "rounds", 50)
        self.k_per_seed = getattr(cfg, "k_per_seed", 64)
        self.top_scaffolds = getattr(cfg, "top_scaffolds", 50)
        self.per_scaffold = getattr(cfg, "per_scaffold", 10)
        self.max_decode_len = getattr(cfg, "max_decode_len", 71)
        self.sigma_start = getattr(cfg, "sigma_start", 0.5)
        self.sigma_end = getattr(cfg, "sigma_end", 0.05)
        self.seed = getattr(cfg, "seed", 42)
        self.elite_fraction = getattr(cfg, "elite_fraction", 0.05)
        self.patience = getattr(cfg, "patience", 3)
        self.tol = getattr(cfg, "tol", 1e-3)
        self.nominal_budget = 100_000

    def _calc_k_per_seed(self, pop, rounds, budget):
        if pop <= 0 or rounds <= 0:
            return 1
        return max(1, budget // (pop * rounds))

    def _planned_total(self, pop, k, rounds):
        return int(pop) * int(k) * int(rounds)

    def run(self, scoring_function, budget_mols: int, starting_population=None):
        benchmark_name = getattr(scoring_function, "name", "")
        mode_seeking = any(
            k in benchmark_name.lower() for k in ["rediscovery", "similarity"]
        )
        logger.info(f"[scaf] mode_seeking={mode_seeking} | benchmark={benchmark_name}")

        if mode_seeking:
            top_scaffolds_init, per_scaffold_init = 10, 10
            sigma_start, sigma_end = 0.08, 0.005
            patience = 5
        else:
            top_scaffolds_init, per_scaffold_init = (
                self.top_scaffolds,
                self.per_scaffold,
            )
            sigma_start, sigma_end = self.sigma_start, self.sigma_end
            patience = self.patience

        k_eff = self.k_per_seed or self._calc_k_per_seed(
            self.population, self.n_rounds, self.nominal_budget
        )
        logger.info(
            f"[scaf] rounds={self.n_rounds}, pop={self.population}, k={k_eff}, σ∈[{sigma_start:.3f},{sigma_end:.3f}]"
        )

        if starting_population:
            logger.info(
                f"[scaf] Using provided starting_population: {len(starting_population)} molecules"
            )
            if hasattr(self.model, "encode_smiles"):
                seeds = self.model.encode_smiles(starting_population)
            else:
                seeds = torch.randn(
                    self.population, self.model.latent_dim, device=self.model.device
                )
        else:
            seeds = torch.randn(
                self.population, self.model.latent_dim, device=self.model.device
            )

        all_generated = []
        sigma_sched = np.linspace(sigma_start, sigma_end, self.n_rounds)
        best_global = -np.inf
        stagnation = 0

        for round_idx in range(self.n_rounds):
            sigma = sigma_sched[round_idx]

            if mode_seeking and round_idx >= 3:
                top_scaffolds = 1
                per_scaffold = 20
                k_local = max(k_eff, 32)
                sigma = max(sigma, 0.005)
            else:
                top_scaffolds = top_scaffolds_init
                per_scaffold = per_scaffold_init
                k_local = k_eff

            logger.info(
                f"[scaf] Round {round_idx+1}/{self.n_rounds}, σ={sigma:.3f}, top_scaffolds={top_scaffolds}, per_scaffold={per_scaffold}"
            )

            smiles_batch = []
            for i, z_seed in enumerate(seeds):
                z_seed = z_seed.unsqueeze(0).repeat(k_local, 1)
                z_batch = z_seed + sigma * torch.randn_like(z_seed)
                z_batch = torch.nan_to_num(z_batch, nan=0.0, posinf=1e2, neginf=-1e2)

                top_k = 20 if round_idx < 2 and not mode_seeking else 1
                smi_list = self.model.decode_latents(
                    z_batch, max_len=self.max_decode_len, top_k=top_k
                )
                smiles_batch.extend(
                    [(i, s) for s in smi_list if s and smiles_is_valid(s)]
                )

            if not smiles_batch:
                logger.warning("[scaf] No valid molecules generated this round.")
                break

            gen_df = pd.DataFrame(smiles_batch, columns=["seed_id", "std_smiles"])
            val_smiles = gen_df["std_smiles"].tolist()

            try:
                scores = score_list(scoring_function, val_smiles)
            except Exception as e:
                logger.error(f"[scaf] score_list() failed: {e}")
                scores = np.zeros(len(val_smiles))

            gen_df["score"] = pd.Series(scores, dtype=float).fillna(0.0)
            gen_df["label_lb"] = -gen_df["score"].astype(float)

            gen_df = compute_scaffolds(
                gen_df,
                smiles_col="std_smiles",
                scaf_col="Murcko Scaffold",
                scaf_id_col="Murcko Scaffold ID",
                scaf_size_col="Murcko Scaffold Cluster Size",
            )

            n_valid = len(gen_df)
            n_unique_scaffolds = gen_df["Murcko Scaffold"].nunique(dropna=True)
            best_score = float(gen_df["score"].max())
            logger.info(
                f"[scaf] stats | valid={n_valid:,} | scaffolds={n_unique_scaffolds:,} | best_score={best_score:.3f}"
            )

            if best_score <= best_global + self.tol:
                stagnation += 1
                if stagnation >= patience:
                    logger.info(f"[scaf] Early stopping after {round_idx+1} rounds.")
                    break
            else:
                best_global = best_score
                stagnation = 0

            min_n = 1 if mode_seeking else 2
            ranked_scaffolds = rank_scaffolds_label(
                gen_df,
                scaf_col="Murcko Scaffold",
                y_col="label_lb",
                min_n=min_n,
            )
            if ranked_scaffolds.empty:
                break

            selected_df = top_diverse_molecules_within_top_scaffolds(
                df=gen_df,
                ranked_scaffolds=ranked_scaffolds,
                scaf_col="Murcko Scaffold",
                y_col="label_lb",
                smiles_col="std_smiles",
                selfies_col=(
                    "canonical_selfies"
                    if "canonical_selfies" in gen_df.columns
                    else None
                ),
                score_col="mean",
                top_scaffolds=top_scaffolds,
                per_scaffold=per_scaffold,
                radius=2,
                nBits=2048,
                seed=round_idx,
            )
            if selected_df.empty:
                break

            next_smiles = selected_df["std_smiles"].tolist()
            next_seeds = self.model.encode_smiles(next_smiles)

            per_seed = gen_df.groupby("seed_id")["score"].max().reset_index()
            elite_k = max(1, int(self.elite_fraction * self.population))
            elite_seed_ids = (
                per_seed.sort_values("score", ascending=False)
                .head(elite_k)["seed_id"]
                .to_numpy()
            )
            elite_idx = torch.as_tensor(
                elite_seed_ids, device=seeds.device, dtype=torch.long
            )
            elite_idx = elite_idx.clamp_(0, seeds.size(0) - 1)
            elite_z = seeds[elite_idx]

            seeds = torch.cat([elite_z, next_seeds.to(seeds.device)], dim=0)
            if seeds.size(0) < self.population:
                repeats = self.population // max(1, seeds.size(0)) + 1
                seeds = seeds.repeat((repeats, 1))[: self.population]
            else:
                seeds = seeds[: self.population]

            all_generated.append(gen_df)

        all_df = (
            pd.concat(all_generated, ignore_index=True)
            if all_generated
            else pd.DataFrame()
        )
        all_df = all_df.drop_duplicates(subset=["std_smiles"]).copy()
        all_df = all_df.sort_values("score", ascending=False)

        logger.info(f"[scaf] Completed {len(all_df):,} unique molecules total.")
        return all_df["std_smiles"].head(budget_mols).tolist()


class EmbeddingCacheBuilderStrategy(LatentStrategy):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.smiles_file = cfg.smiles_file
        self.batch_size = getattr(cfg, "batch_size", 1024)
        self.chunk_size = getattr(cfg, "chunk_size", 50_000)
        self.out_dir = os.path.join("cache", "embeddings")
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_path = os.path.join(self.out_dir, "embeddings.parquet")

    def _read_smiles_chunks(self):
        """Yield canonical, valid SMILES in chunks."""
        if not self.smiles_file or not os.path.exists(self.smiles_file):
            raise FileNotFoundError(f"Missing SMILES file: {self.smiles_file}")

        buf = []
        with open(self.smiles_file) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    s = canonicalize(s)
                    if s and smiles_is_valid(s):
                        buf.append(s)
                except Exception:
                    continue
                if len(buf) >= self.chunk_size:
                    yield buf
                    buf = []
        if buf:
            yield buf

    def _encode_batch(self, smiles_batch):
        self.model.model.eval()
        latents = []
        with torch.no_grad():
            for i in range(0, len(smiles_batch), self.batch_size):
                sub = smiles_batch[i : i + self.batch_size]
                try:
                    lat = self.model.encode_smiles(sub)
                    latents.append(lat.detach().cpu())
                except Exception as e:
                    logger.warning(f"[embed] batch failed: {e}")
                    latents.append(torch.randn(len(sub), self.model.latent_dim))
        return torch.cat(latents, dim=0)

    def run(self, scoring_function=None, budget_mols=None):

        if os.path.exists(self.out_path):
            logger.info(f"[embed] Found existing embeddings → {self.out_path}")
            logger.info("[embed] Skipping recomputation.")
            return []

        logger.info(f"[embed] Building embeddings → {self.out_path}")
        logger.info(f"[embed] Reading SMILES from {self.smiles_file}")

        writer = None
        total = 0
        chunk_iter = list(self._read_smiles_chunks())

        with tqdm(total=len(chunk_iter), desc="[embed] chunks", ncols=100) as pbar:
            for chunk_idx, smiles_chunk in enumerate(chunk_iter):
                pbar.set_postfix_str(f"{len(smiles_chunk):,} SMILES")
                latents = self._encode_batch(smiles_chunk)
                latent_cols = [f"z_{j}" for j in range(latents.shape[1])]

                df = pd.DataFrame(
                    {
                        "smiles": smiles_chunk,
                        **{
                            col: latents[:, j].numpy()
                            for j, col in enumerate(latent_cols)
                        },
                    }
                )

                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        self.out_path, table.schema, compression="zstd"
                    )
                writer.write_table(table)

                total += len(df)
                pbar.update(1)
                pbar.set_postfix_str(f"total={total:,}")

        if writer:
            writer.close()
            logger.info(f"[embed] Done. {total:,} embeddings written → {self.out_path}")
        return []


class ScoreCacheStrategy(LatentStrategy):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.embedding_file = getattr(
            cfg, "embedding_file", "cache/embeddings/embeddings.parquet"
        )
        self.out_dir_root = "cache"
        os.makedirs(self.out_dir_root, exist_ok=True)

    def run(self, scoring_function, budget_mols: int):

        if not os.path.exists(self.embedding_file):
            logger.error(f"[score] Missing embeddings file: {self.embedding_file}")
            return []

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.out_dir_root, timestamp)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "scores.parquet")

        logger.info(f"[score] Loading embeddings from {self.embedding_file}")
        table = pq.read_table(self.embedding_file)
        df = table.to_pandas()

        smiles = df["smiles"].tolist()
        from tqdm import tqdm

        scores = []
        for i in tqdm(range(0, len(smiles), 1000), desc="[score] scoring"):
            sub = smiles[i : i + 1000]
            try:
                scores.extend(score_list(scoring_function, sub))
            except Exception as e:
                logger.warning(f"[score] batch failed: {e}")
                scores.extend([0.0] * len(sub))

        df["score"] = scores
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False), out_path, compression="zstd"
        )
        logger.info(f"[score] Done. Wrote {len(df):,} rows → {out_path}")
        return smiles[:100]


class VAEGoalDirectedGenerator(GoalDirectedGenerator):
    def __init__(
        self,
        model_wrapper: "VAEModelWrapper",
        strategy: str = "evo",
        cfg: Optional["StrategyConfig"] = None,
    ):
        self.model = model_wrapper
        self.cfg = cfg or StrategyConfig()
        self.strategy_name = strategy.lower()

        self._strategies: Dict[str, "LatentStrategy"] = {
            "random": RandomSamplingStrategy(self.model, self.cfg),
            "hill": HillClimbStrategy(self.model, self.cfg),
            "evo": EvolutionaryStrategy(self.model, self.cfg),
            "sa": SimulatedAnnealingStrategy(self.model, self.cfg),
            "cma": CMAESStrategy(self.model, self.cfg),
            "scaf": ScaffoldFrontierStrategy(self.model, self.cfg),
            "ga": LatentGAStrategy(self.model, self.cfg),
        }
        self._strategies.update(
            {
                "cache_embed": EmbeddingCacheBuilderStrategy(self.model, self.cfg),
                "cache_score": ScoreCacheStrategy(self.model, self.cfg),
            }
        )

    def generate_optimized_molecules(
        self, scoring_function: Any, number_molecules: int, **kwargs
    ) -> List[str]:
        starting_pop = kwargs.get("starting_population")
        if starting_pop:
            logger.info(
                "[adapter] received starting_population: %d SMILES", len(starting_pop)
            )

        logger.info(
            "[adapter] strategy=%s, target_topN=%d",
            self.strategy_name,
            number_molecules,
        )

        strat = self._strategies.get(self.strategy_name)
        if strat is None:
            raise ValueError(
                f"Unknown strategy '{self.strategy_name}'. "
                f"Choose from {list(self._strategies.keys())}"
            )

        return strat.run(scoring_function, number_molecules)


def run_all_benchmarks(
    model_wrapper: "VAEModelWrapper",
    strategy: str = "evo",
    cfg: Optional["StrategyConfig"] = None,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", strategy)
    os.makedirs(out_dir, exist_ok=True)

    json_file = os.path.join(out_dir, f"goal_directed_{strategy}_{timestamp}.json")
    cfg_file = os.path.join(out_dir, f"config_{strategy}_{timestamp}.json")

    logger.info(f"[runner] Running strategy={strategy}, results → {json_file}")

    if cfg is not None:
        try:
            with open(cfg_file, "w") as f:
                json.dump(cfg.__dict__, f, indent=2)
            logger.info(f"[runner] Config written to {cfg_file}")
        except Exception as e:
            logger.warning(f"[runner] Could not write config file: {e}")

    gen = VAEGoalDirectedGenerator(model_wrapper, strategy=strategy, cfg=cfg)
    assess_goal_directed_generation(
        goal_directed_molecule_generator=gen,
        json_output_file=json_file,
        benchmark_version="v2",
    )

    logger.info(f"[runner] Benchmark complete → {json_file}")
    return json_file


def main(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logging.info(f"Using device: {device}")

    lm = VAELightningModule.load_from_checkpoint(
        args.ckpt_path,
        tokenizer="bmfm_sm_gen.data.base_tokenizer.MolecularTokenizer",
        vocab_file=args.vocab_file,
        map_location=device,
    )
    lm.to(device)
    lm.eval()
    lm.model.eval()

    tokenizer = SELFIESTokenizer(vocab=args.vocab_file)
    wrapper = VAEModelWrapper(lm.model, tokenizer, latent_dim=lm.model.latent_dim)

    base_cfg = StrategyConfig(
        population=args.population,
        rounds=args.rounds,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        k_per_seed=args.k_per_seed,
        top_scaffolds=args.top_scaffolds,
        per_scaffold=args.per_scaffold,
        max_decode_len=args.max_decode_len,
        seed=args.seed,
        log_every=args.log_every,
        smiles_file=args.smiles_file,
    )

    if not args.sweep:
        results = run_all_benchmarks(wrapper, strategy=args.strategy, cfg=base_cfg)
        print(results)
        return

    sweep_cfgs: List[Tuple[str, StrategyConfig]] = []
    strat = args.strategy.lower()
    if strat == "hill":
        for pop in [128, 256]:
            for s0 in [0.10, 0.15]:
                sweep_cfgs.append(
                    (
                        "hill",
                        dataclasses.replace(base_cfg, population=pop, sigma_start=s0),
                    )
                )
    elif strat == "evo":
        for pop in [128, 256]:
            for surv in [32, 64]:
                sweep_cfgs.append(
                    (
                        "evo",
                        dataclasses.replace(base_cfg, population=pop, survivors=surv),
                    )
                )
    elif strat == "sa":
        for s0 in [0.10, 0.15]:
            sweep_cfgs.append(("sa", dataclasses.replace(base_cfg, sigma_start=s0)))
    elif strat == "cma":
        for pop in [64, 128]:
            sweep_cfgs.append(("cma", dataclasses.replace(base_cfg, population=pop)))
    elif strat == "scaf":
        for kps in [8, 16]:
            for top_s in [10, 20]:
                sweep_cfgs.append(
                    (
                        "scaf",
                        dataclasses.replace(
                            base_cfg, k_per_seed=kps, top_scaffolds=top_s
                        ),
                    )
                )
    elif strat == "random":
        for pop in [256, 512]:
            sweep_cfgs.append(("random", dataclasses.replace(base_cfg, population=pop)))
    else:
        sweep_cfgs.append((strat, base_cfg))

    for name, cfg in sweep_cfgs:
        logger.info(f"[sweep] strategy={name} cfg={cfg}")
        res = run_all_benchmarks(wrapper, strategy=name, cfg=cfg)
        print(f"[sweep:{name}] {res}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="GuacaMol Goal-Directed Benchmark with VAE"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="/path/to/vae-epochepoch=06-step=260496-val_loss=0.29.ckpt",
        help="Path to Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        default="/path/to/vocab.json",
        help="Path to tokenizer vocab.json",
    )

    # Search parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="scaf",
        choices=[
            "random",
            "hill",
            "evo",
            "sa",
            "cma",
            "scaf",
            "ga",
            "cache_embed",
            "cache_score",
        ],
    )
    parser.add_argument(
        "--population", type=int, default=512, help="Latent population / seeds"
    )
    parser.add_argument("--rounds", type=int, default=50, help="Number of iterations")
    parser.add_argument(
        "--sigma_start",
        type=float,
        default=0.5,
        help="Latent perturbation start scale",
    )
    parser.add_argument(
        "--sigma_end", type=float, default=0.05, help="Latent perturbation end scale"
    )
    parser.add_argument(
        "--k_per_seed", type=int, default=64, help="Candidates per seed (scaf)"
    )
    parser.add_argument(
        "--top_scaffolds", type=int, default=50, help="Top scaffolds per round (scaf)"
    )
    parser.add_argument(
        "--per_scaffold", type=int, default=20, help="Molecules per scaffold (scaf)"
    )
    parser.add_argument(
        "--max_decode_len", type=int, default=71, help="Maximum decode length"
    )
    parser.add_argument("--smiles_file", default="data/guacamol_v1_all.smiles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_every", type=int, default=2, help="Logging cadence (rounds)"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a tiny parameter sweep for the chosen strategy",
    )
    args = parser.parse_args()
    main(args)
