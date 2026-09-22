import pandas as pd
import altair as alt
import os
import numpy as np
from rdkit import Chem
from typing import Optional, List, Literal
import selfies as sf
import logging
import pathlib
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
import tempfile
import shutil
import json
import math
from bmfm_sm_gen.utils.data_split import *

alt.data_transformers.disable_max_rows()

def get_penultimate_tokens(paths: list[str]) -> list[str]:
    penultimate_tokens = []
    
    for path in paths:
        tokens = [token for token in path.split(os.sep) if token]
        
        if len(tokens) >= 2:
            penultimate_tokens.append(tokens[-2])
            
    return penultimate_tokens

def find_parquet_files(
    base_dir: str = 'cacheNew', 
    file_pattern: str = '**/scores.parquet'
) -> List[str]:
    
    base_path = pathlib.Path(base_dir)
    file_paths = [str(p) for p in base_path.glob(file_pattern)]    
    return file_paths

def smiles_to_selfies(smiles_str: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles_str}")
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        selfies_str = sf.encoder(canonical_smiles)
        return selfies_str
    except Exception as e:
        logging.error(f"Error converting SMILES to SELFIES: {e}")
        return None

def add_selfies_column(
    df: pd.DataFrame,
    smiles_col: str = "std_smiles",
    selfies_col: str = "canonical_selfies",
    drop_failed: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out[selfies_col] = out[smiles_col].apply(smiles_to_selfies)
    if drop_failed:
        before = len(out)
        out = out[~out[selfies_col].isna()].copy()
        dropped = before - len(out)
        if dropped:
            logging.info(f"[add_selfies_column] Dropped {dropped} rows with invalid/un-encodable SMILES.")
    return out

def apply_local_scaling(
    df: pd.DataFrame, 
    scaling_scheme: Literal['minmax', 'zscore', 'robust'], 
    score_col: str = 'score'
) -> pd.DataFrame:
    
    scores = df[score_col]
    
    local_min = scores.min()
    local_max = scores.max()
    local_mean = scores.mean()
    local_std = scores.std()
    local_median = scores.median()
    local_q1 = scores.quantile(0.25)
    local_q3 = scores.quantile(0.75)
    
    new_col_name = f'y_normalized_{scaling_scheme}'
    
    if scaling_scheme == 'minmax':
        df[new_col_name] = (scores - local_min) / (local_max - local_min)
        
    elif scaling_scheme == 'zscore':
        df[new_col_name] = (scores - local_mean) / local_std
        
    elif scaling_scheme == 'robust':
        IQR = local_q3 - local_q1
        df[new_col_name] = (scores - local_median) / IQR
        
    return df

def normalize_data(task_files: List[str], output_dir: str):
    scaling_schemes: List[Literal['minmax', 'zscore', 'robust']] = ['minmax', 'zscore', 'robust']    
    os.makedirs(output_dir, exist_ok=True)
    output_base_names = get_penultimate_tokens(task_files)
    for i, file_path in enumerate(task_files):        
        df = load_smiles_score_from_parquet(
            file_path=file_path,
            smiles_col='smiles',
            score_col='score'
        )
        
        for scheme in scaling_schemes:
            df = apply_local_scaling(
                df=df,
                scaling_scheme=scheme, 
                score_col='score'
            )        
        output_filename = f"{output_base_names[i]}.csv"
        output_path = os.path.join(output_dir, output_filename)        
        df.to_csv(output_path, index=False)
        print(f"Successfully wrote normalized data to: {output_path}")

def find_csv_files(input_dir: str) -> List[str]:
    base_path = pathlib.Path(input_dir)
    file_paths = [str(p) for p in base_path.glob('**/*.csv')]    
    return file_paths
    
def split_data(input_dir: str, output_dir: str, val_size: int = 50000):
    os.makedirs(output_dir, exist_ok=True)
    task_files = find_csv_files(input_dir)  
    for file_path in task_files:
        print(f"Working on {file_path}")
        df = pd.read_csv(file_path)
        df['selfies'] = df['smiles'].progress_apply(smiles_to_selfies)
        num_rows = len(df)
        train_size = num_rows - val_size
        
        base_filename_with_ext = os.path.basename(file_path)
        base_name = os.path.splitext(base_filename_with_ext)[0]
        
        df_shuffled = df.sample(frac=1, random_state=42)
        
        df_train = df_shuffled.iloc[:train_size]
        df_val = df_shuffled.iloc[train_size:]
        
        train_path = os.path.join(output_dir, f"{base_name}_train.csv")
        val_path = os.path.join(output_dir, f"{base_name}_val.csv")
        
        df_train.to_csv(train_path, index=False)
        df_val.to_csv(val_path, index=False)

def update_val_csvs(input_dir: str, add_rdkit_fp: bool = False, radius: int = 2, nBits: int = 2048):
    input_dir = os.path.abspath(input_dir)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith("_val.csv")]
    if not csv_files:
        print(f"No *_val.csv files found in {input_dir}")
        return

    for fname in csv_files:
        fpath = os.path.join(input_dir, fname)
        print(f"Processing {fpath} ...")
        df = pd.read_csv(fpath)

        scaffolds, morgan_fps = [], []
        rdkit_fps = [] if add_rdkit_fp else None

        for smi in df["smiles"]:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                scaffolds.append("")
                morgan_fps.append("")
                if add_rdkit_fp:
                    rdkit_fps.append("")
                continue

            scaf = MurckoScaffold.GetScaffoldForMol(m)
            scaffolds.append(Chem.MolToSmiles(scaf, isomericSmiles=False))

            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
            arr = np.zeros((nBits,), dtype=np.uint8)
            ConvertToNumpyArray(fp, arr)
            morgan_fps.append(json.dumps(arr.tolist()))

            if add_rdkit_fp:
                fp2 = RDKFingerprint(m, fpSize=nBits)
                arr2 = np.zeros((nBits,), dtype=np.uint8)
                ConvertToNumpyArray(fp2, arr2)
                rdkit_fps.append(json.dumps(arr2.tolist()))

        df["scaffold"] = scaffolds
        df["morgan_fp"] = morgan_fps
        if add_rdkit_fp:
            df["rdkit_fp"] = rdkit_fps

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=input_dir)
        os.close(tmp_fd)
        df.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, fpath)
        print(f" → Overwritten {fpath}")


def add_scaffold_column_to_val_csvs(input_dir: str):
    input_dir = os.path.abspath(input_dir)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith("_val.csv")]
    for fname in csv_files:
        fpath = os.path.join(input_dir, fname)
        print(f"Processing {fpath} ...")
        df = pd.read_csv(fpath)
        if "scaffold" in df.columns:
            continue
        scaffolds = []
        for smi in df["smiles"]:
            m = Chem.MolFromSmiles(smi)
            if m:
                scaf = MurckoScaffold.GetScaffoldForMol(m)
                scaffolds.append(Chem.MolToSmiles(scaf, isomericSmiles=False))
            else:
                scaffolds.append("")
        df["scaffold"] = scaffolds
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", dir=input_dir)
        os.close(tmp_fd)
        df.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, fpath)


def main():
    radius = 2
    nBits = 2048
    scaling_schemes = ['minmax', 'zscore', 'robust']
    df_dict = {}
    
    input_dir = os.environ.get('CHEMBL_DATA_DIR', 'data/chembl/clean')
    output_path = os.environ.get('CHEMBL_OUTPUT_DIR', 'data/chembl/output')
    
    os.makedirs(output_path, exist_ok=True)
    
    for path in os.listdir(input_dir):
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)   
        df_dict[name] = pd.read_csv(os.path.join(input_dir, path), index_col=False)
        # 1. remove NaN
        df_dict[name] = df_dict[name][~df_dict[name]['standard_value_nm'].isna()]
        # 2. log base 10: standard_value_nm -> standard_value_nm_log_10
        df_dict[name]['standard_value_nm_log_10'] = np.log10(df_dict[name]['standard_value_nm'])
        # 3. add selfies column: std_smiles -> canonical_selfies
        df_dict[name] = add_selfies_column( df_dict[name], smiles_col="std_smiles", selfies_col="canonical_selfies", drop_failed=True)
        # 4. add normalization: standard_value_nm_log_10 -> *_minmax, *_zscore, *_robust
        for scheme in scaling_schemes:
            df_dict[name] = apply_local_scaling(
                df=df_dict[name],
                scaling_scheme=scheme, 
                score_col='standard_value_nm_log_10'
            )
        # 5. add scaffolds, morgan & rdkit fingerprints: scaffold, morgan_fp, rdkit_fp
        scaffolds, morgan_fps, rdkit_fps = [], [], []

        for smi in df_dict[name]["std_smiles"]:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                scaffolds.append("")
                morgan_fps.append("")
                rdkit_fps.append("")
                continue

            scaf = MurckoScaffold.GetScaffoldForMol(m)
            scaffolds.append(Chem.MolToSmiles(scaf, isomericSmiles=False))

            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
            arr = np.zeros((nBits,), dtype=np.uint8)
            ConvertToNumpyArray(fp, arr)
            morgan_fps.append(json.dumps(arr.tolist()))

            fp2 = RDKFingerprint(m, fpSize=nBits)
            arr2 = np.zeros((nBits,), dtype=np.uint8)
            ConvertToNumpyArray(fp2, arr2)
            rdkit_fps.append(json.dumps(arr2.tolist()))

        df_dict[name]["scaffold"] = scaffolds
        df_dict[name]["morgan_fp"] = morgan_fps
        df_dict[name]["rdkit_fp"] = rdkit_fps
        
        # 6. split into train, validation, and test by time: *_train.csv, *_val.csv, *_test.csv
        df_train, df_val, df_test, _ = split_by_year(df_dict[name],year_col='New Document Year')
        df_train.to_csv(os.path.join(output_path, f'{name}_train.csv'), index=False)
        df_val.to_csv(os.path.join(output_path, f'{name}_val.csv'), index=False)
        df_test.to_csv(os.path.join(output_path, f'{name}_test.csv'), index=False)
    
if __name__ == "__main__":
    main()