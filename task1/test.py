import pandas as pd
import networkx as nx
import numpy as np
from skfp.model_selection import scaffold_train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from skfp.preprocessing import MolFromSmilesTransformer, MolStandardizer
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint, TopologicalTorsionFingerprint
from sklearn.metrics import f1_score


# ==========================================
# 1. FUNKCJE POMOCNICZE
# ==========================================

def build_hierarchy_map(obo_path, column_names):
    """
    Tworzy mapę relacji rodzic-dziecko opartą na indeksach kolumn.
    column_names: lista nazw kolumn z etykietami (bez SMILES i ID).
    """
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

    # Sortowanie topologiczne (od szczegółu do ogółu)
    sorted_indices = list(nx.topological_sort(dag))[::-1]

    parent_child_map = {}
    for child in range(len(column_names)):
        parents = list(dag.predecessors(child))
        if parents:
            parent_child_map[child] = parents

    return parent_child_map, sorted_indices


def calculate_inconsistencies(y_probs, parent_child_map):
    """
    Oblicza średnią liczbę niespójności na próbkę.
    Niespójność to sytuacja, gdy P(dziecko) > P(rodzic).
    """
    inconsistencies = 0
    n_samples = y_probs.shape[0]

    for child_idx, parents in parent_child_map.items():
        for parent_idx in parents:
            violations = np.sum(y_probs[:, child_idx] > y_probs[:, parent_idx] + 1e-6)
            inconsistencies += violations

    return inconsistencies / n_samples


def fix_hierarchy_consistency(y_probs, parent_child_map, sorted_classes):
    """
    Naprawia prawdopodobieństwa od dołu do góry (od liści do korzenia).
    """
    y_consistent = y_probs.copy()

    for child_idx in sorted_classes:
        if child_idx in parent_child_map:
            for parent_idx in parent_child_map[child_idx]:
                # Rodzic musi mieć P >= dziecko
                y_consistent[:, parent_idx] = np.maximum(
                    y_consistent[:, parent_idx],
                    y_consistent[:, child_idx]
                )
    return y_consistent


def evaluate_model(pipeline, X_test, y_test, parent_child_map, sorted_idx):
    print("Generowanie predykcji (to może chwilę potrwać)...")

    # 1. Pobranie prawdopodobieństw z modelu
    probs_list = pipeline.predict_proba(X_test)

    # 2. Przekształcenie listy na macierz (n_samples, 500_classes)
    y_probs = np.column_stack([p[:, 1] for p in probs_list])

    # 3. Sprawdzenie niespójności PRZED naprawą
    inconsistencies_before = calculate_inconsistencies(y_probs, parent_child_map)
    print(f"Średnia liczba niespójności na cząsteczkę (przed naprawą): {inconsistencies_before:.4f}")

    # 4. Naprawa hierarchii
    print("Naprawianie spójności hierarchii...")
    y_probs_consistent = fix_hierarchy_consistency(y_probs, parent_child_map, sorted_idx)

    # 5. Sprawdzenie niespójności PO naprawie (powinno być równe 0.0)
    inconsistencies_after = calculate_inconsistencies(y_probs_consistent, parent_child_map)
    print(f"Średnia liczba niespójności na cząsteczkę (po naprawie): {inconsistencies_after:.4f}")

    # 6. Binaryzacja predykcji (standardowy próg 0.5)
    y_pred_bin = (y_probs_consistent >= 0.5).astype(int)

    # 7. Obliczenie końcowego F1 Macro
    macro_f1 = f1_score(y_test, y_pred_bin, average='macro')
    print(f"\n====================================")
    print(f"⭐ Twój Macro-averaged F1 Score: {macro_f1:.4f} ⭐")
    print(f"====================================")

    return y_probs_consistent


# ==========================================
# 2. ŁADOWANIE I PODZIAŁ DANYCH
# ==========================================

train_df = pd.read_parquet('chebi_dataset_train.parquet')

smiles = train_df["SMILES"].values
labels_columns = train_df.drop(columns=["SMILES", "mol_id"]).columns
y = train_df[labels_columns].values

X_train, X_test, y_train, y_test = scaffold_train_test_split(
    smiles,
    y,
    test_size=0.2,
)

print(f"Liczba cząsteczek w treningu: {len(X_train)}")
print(f"Liczba cząsteczek w teście: {len(X_test)}")


# ==========================================
# 3. BUDOWA MAPY HIERARCHII
# ==========================================

pc_map, sorted_idx = build_hierarchy_map('chebi_classes.obo', labels_columns)


# ==========================================
# 4. BUDOWA I TRENING MODELU
# ==========================================

fps_union = FeatureUnion([
    ("ecfp", ECFPFingerprint(n_jobs=-1)),
    ("maccs", MACCSFingerprint(n_jobs=-1)),
    ("tt", TopologicalTorsionFingerprint(n_jobs=-1))
])

lgbm = LGBMClassifier(class_weight="balanced", n_jobs=-1, random_state=0, verbose=-1)

pipeline = Pipeline([
    ("mol_from_smiles", MolFromSmilesTransformer()),
    ("mol_standardizer", MolStandardizer()),
    ("fps_union", fps_union),
    ("classifier", MultiOutputClassifier(lgbm))
])

print("Rozpoczynam trenowanie modelu...")
pipeline.fit(X_train, y_train)
print("Trenowanie zakończone!")


# ==========================================
# 5. EWALUACJA
# ==========================================

y_test_probs = evaluate_model(pipeline, X_test, y_test, pc_map, sorted_idx)