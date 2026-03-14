import os
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem

# IMPORTY DO MODELU
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from skfp.preprocessing import MolFromSmilesTransformer, MolStandardizer
from skfp.fingerprints import (
    ECFPFingerprint, 
    MACCSFingerprint, 
    TopologicalTorsionFingerprint,
    EStateFingerprint  
)

TRAIN_FILE = "chebi_dataset_train.parquet"
TEST_FILE = "chebi_submission_example.parquet"
OBO_FILE = "chebi_classes.obo"

# Zdefiniuj nazwę pliku wyjściowego
OUTPUT_PARQUET = "moje_predykcje.parquet"
OUTPUT_CSV = "moje_predykcje.csv" # Opcjonalnie, do łatwego podglądu


# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================

def build_hierarchy_map(obo_path, column_names):
    id_to_idx = {name: i for i, name in enumerate(column_names)}

    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(column_names)))

    current_id = None
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('id:'):
                current_id = line.split('id: ')[1]
            elif line.startswith('is_a:') and current_id in id_to_idx:
                parent_id = line.split('is_a: ')[1].split(' ! ')[0]
                if parent_id in id_to_idx:
                    dag.add_edge(id_to_idx[parent_id], id_to_idx[current_id])

    sorted_indices = list(nx.topological_sort(dag))[::-1]

    parent_child_map = {}
    for child in range(len(column_names)):
        parents = list(dag.predecessors(child))
        if parents:
            parent_child_map[child] = parents

    return parent_child_map, sorted_indices


def fix_hierarchy_consistency(y_probs, parent_child_map, sorted_classes):
    y_consistent = y_probs.copy()

    for child_idx in sorted_classes:
        if child_idx in parent_child_map:
            for parent_idx in parent_child_map[child_idx]:
                y_consistent[:, parent_idx] = np.maximum(
                    y_consistent[:, parent_idx],
                    y_consistent[:, child_idx]
                )
    return y_consistent


def is_valid_smiles(s):
    try:
        m = Chem.MolFromSmiles(s)
        return m is not None
    except:
        return False


# ==========================================
# GŁÓWNA LOGIKA
# ==========================================

def main():
    # 1. Wczytanie i filtrowanie danych treningowych
    print("Wczytywanie danych treningowych...")
    train_df = pd.read_parquet(TRAIN_FILE)

    print("Sprawdzanie poprawności SMILES...")
    initial_count = len(train_df)
    train_df = train_df[train_df["SMILES"].apply(is_valid_smiles)]
    final_count = len(train_df)
    if initial_count != final_count:
        print(f"Usunięto {initial_count - final_count} niepoprawnych kodów SMILES.")

    smiles_train = train_df["SMILES"].values
    labels_columns = train_df.drop(columns=["SMILES", "mol_id"]).columns
    y_train = train_df[labels_columns].values

    # 2. Budowa mapy hierarchii
    print("Budowanie mapy hierarchii...")
    pc_map, sorted_idx = build_hierarchy_map(OBO_FILE, labels_columns)

    # 3. Budowa i trening modelu
    fps_union = FeatureUnion([
        ("ecfp", ECFPFingerprint(fp_size=2048, radius=3, count=True, n_jobs=-1)),
        ("maccs", MACCSFingerprint(n_jobs=-1)),
        ("tt", TopologicalTorsionFingerprint(n_jobs=-1)),
        ("estate", EStateFingerprint(n_jobs=-1))
    ])

    lgbm = LGBMClassifier(
        class_weight="balanced",
        n_jobs=1,          
        random_state=0,
        verbose=-1,
        n_estimators=500,
        max_depth=6,
        num_leaves=31,
    )

    pipeline = Pipeline([
        ("mol_from_smiles", MolFromSmilesTransformer(n_jobs=-1)),
        ("mol_standardizer", MolStandardizer(n_jobs=-1)),
        ("fps_union", fps_union),
        ("scaler", MinMaxScaler()), 
        ("classifier", MultiOutputClassifier(lgbm, n_jobs=-1)), 
    ])

    print(f"Trenowanie modelu równolegle na {os.cpu_count()} rdzeniach...")
    pipeline.fit(smiles_train, y_train)
    print("Trenowanie zakończone!")

    # 4. Wczytanie danych do predykcji
    try:
        submission_df = pd.read_parquet(TEST_FILE)
    except Exception as e:
        raise FileExistsError(f"Nie udało się wczytać pliku Parquet z testami, błąd: {e}")

    smiles_test = submission_df["SMILES"].values

    # 5. Generowanie predykcji
    print("Generowanie predykcji...")
    probs_list = pipeline.predict_proba(smiles_test)
    y_probs = np.column_stack([p[:, 1] for p in probs_list])

    # 6. Naprawa spójności hierarchii
    print("Naprawianie spójności hierarchii...")
    y_probs_consistent = fix_hierarchy_consistency(y_probs, pc_map, sorted_idx)

    # 7. Binaryzacja i przypisanie wyników do DataFrame
    y_pred_bin = (y_probs_consistent >= 0.5).astype(int)
    for i, col in enumerate(labels_columns):
        submission_df[col] = y_pred_bin[:, i]

    print("Przykładowe predykcje:")
    print(submission_df.head())

    # 8. Zapis wyników do pliku lokalnego zamiast wysyłania na serwer
    print(f"Zapisywanie predykcji do pliku {OUTPUT_PARQUET}...")
    submission_df.to_parquet(OUTPUT_PARQUET, index=False)
    
    print(f"Zapisywanie predykcji do pliku {OUTPUT_CSV} (do podglądu)...")
    submission_df.to_csv(OUTPUT_CSV, index=False)

    print("Gotowe! Wszystkie pliki zostały zapisane lokalnie.")


if __name__ == "__main__":
    main()