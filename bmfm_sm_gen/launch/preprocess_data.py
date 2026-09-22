import argparse
import logging

import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm


def smiles_to_selfies(smiles_str: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles_str}")
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        selfies_str = sf.encoder(canonical_smiles)
        return selfies_str
    except Exception as e:
        logging.error(f"Error converting SMILES to SELFIES: {e}")
        return None


def calculate_lipinski_properties(smiles_str: str) -> dict:
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return None
        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logP": Descriptors.MolLogP(mol),
            "num_h_donors": Descriptors.NumHDonors(mol),
            "num_h_acceptors": Descriptors.NumHAcceptors(mol),
        }
    except Exception as e:
        logging.error(f"Error calculating Lipinski properties: {e}")
        return None


def convert_smiles_file_to_selfies(input_file: str, output_file: str):
    df = pd.read_csv(input_file)
    selfies_list = []
    lipinski_data = []
    selfies_lengths = []

    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing SMILES"
    ):
        smiles = row["smiles"]
        selfies = smiles_to_selfies(smiles)
        selfies_list.append(selfies)
        
        if selfies is not None:
            selfies_length = len(list(sf.split_selfies(selfies)))
        else:
            selfies_length = None
        selfies_lengths.append(selfies_length)

        lipinski = calculate_lipinski_properties(smiles)
        lipinski_data.append(lipinski)

    df["selfies"] = selfies_list
    df["selfie_length"] = selfies_lengths
    df["molecular_weight"] = [lp["molecular_weight"] if lp else None for lp in lipinski_data]
    df["logP"] = [lp["logP"] if lp else None for lp in lipinski_data]
    df["num_h_donors"] = [lp["num_h_donors"] if lp else None for lp in lipinski_data]
    df["num_h_acceptors"] = [lp["num_h_acceptors"] if lp else None for lp in lipinski_data]

    df = df.dropna(subset=["selfies"])

    df.to_csv(output_file, index=False)
    logging.info(f"Processing complete. Updated data saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process SMILES to SELFIES and calculate properties")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input CSV file containing SMILES strings",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output CSV file to save processed data",
    )
    args = parser.parse_args()

    convert_smiles_file_to_selfies(args.input_file, args.output_file)


if __name__ == "__main__":
    main()