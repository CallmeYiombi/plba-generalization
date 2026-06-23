"""Feature extraction and the shared FeatureStore.

Protein features: amino-acid composition (20-dim) and ESM-2 mean-pooled
embeddings (1280-dim). Ligand features: Morgan fingerprints (2048-dim) and
molecular graphs (for GraphDTA). All lookups live in a FeatureStore that the
models read from.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_VOCAB = {aa: i + 1 for i, aa in enumerate(AA_LIST)}
SMILES_VOCAB = {c: i + 1 for i, c in enumerate(
    "#%()+-./0123456789=@BCFIKLMNOPRSZ[\\]abcdegilnoprstu")}

# DeepDTA / GraphDTA encoding lengths (99th percentile of this dataset).
MAX_PROT_LEN = 2734
MAX_SMI_LEN = 601


@dataclass
class FeatureStore:
    uid_to_seq: dict
    uid_to_aac: dict = field(default_factory=dict)
    uid_to_esm2: dict = field(default_factory=dict)
    smiles_to_fp: dict = field(default_factory=dict)
    all_smiles_graphs: dict = field(default_factory=dict)
    failed_smiles: set = field(default_factory=set)
    failed_graph_smiles: set = field(default_factory=set)


def encode_seq(seq, vocab, max_len):
    enc = [vocab.get(c, 0) for c in seq[:max_len]]
    return enc + [0] * (max_len - len(enc))


def seq_to_aa_composition(seq):
    seq = seq.upper()
    if not seq:
        return np.zeros(20)
    return np.array([seq.count(aa) / len(seq) for aa in AA_LIST])


def build_aac(store):
    store.uid_to_aac = {uid: seq_to_aa_composition(seq)
                        for uid, seq in store.uid_to_seq.items()}
    print(f"AAC computed: {len(store.uid_to_aac):,} proteins")


def smiles_to_morgan_fp(smiles, store, radius=2, nbits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            store.failed_smiles.add(smiles)
            return np.zeros(nbits)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
        return np.array(gen.GetFingerprintAsNumPy(mol))
    except Exception:
        store.failed_smiles.add(smiles)
        return np.zeros(nbits)


def build_morgan(smiles_list, store):
    store.smiles_to_fp = {smi: smiles_to_morgan_fp(smi, store) for smi in smiles_list}
    print(f"Morgan FPs computed: {len(store.smiles_to_fp):,} "
          f"(failed: {len(store.failed_smiles):,})")


def smiles_to_graph(smiles):
    """SMILES -> PyG Data; atom feature is 5-dim. Returns None on failure."""
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    feats = [[atom.GetAtomicNum() % 12, atom.GetDegree(), atom.GetTotalNumHs(),
              atom.GetFormalCharge(), int(atom.GetIsAromatic())]
             for atom in mol.GetAtoms()]
    x = torch.tensor(feats, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [(i, j), (j, i)]
    if not edges:  # single atom: self-loop
        edges = [(0, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def build_graphs(smiles_list, store):
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is not None:
            store.all_smiles_graphs[smi] = g
        else:
            store.failed_graph_smiles.add(smi)
    n_ok, n_fail = len(store.all_smiles_graphs), len(store.failed_graph_smiles)
    print(f"Ligand graphs: {n_ok:,} ok / {n_fail:,} failed "
          f"(failed excluded from GraphDTA)")


def load_or_extract_esm2(store, esm2_path, needed_uids, model_path,
                         checkpoint_every=100):
    """Load cached ESM-2 embeddings or extract missing ones (full sequence,
    mean-pooled, FP16). ESM2-650M is trained to 1024 aa; longer proteins
    (~1.9% here) are still processed in full without truncation."""
    if esm2_path.exists():
        store.uid_to_esm2 = np.load(esm2_path, allow_pickle=True).item()
        need_extract = [uid for uid in needed_uids if uid not in store.uid_to_esm2]
        print(f"Loaded ESM-2 cache: {len(store.uid_to_esm2):,} "
              f"({len(need_extract):,} to extract)")
    else:
        need_extract = list(store.uid_to_seq.keys())
        print(f"No ESM-2 cache; extracting {len(need_extract):,}")

    if not need_extract:
        return

    import esm

    orig_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = patched_load
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    torch.load = orig_load

    esm_model = esm_model.to(DEVICE).eval().half()
    batch_converter = alphabet.get_batch_converter()

    def extract_one(seq):
        seq = seq.upper().replace("*", "").replace("U", "X").replace("O", "X")
        _, _, tokens = batch_converter([("seq", seq)])
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            results = esm_model(tokens, repr_layers=[33], return_contacts=False)
        reps = results["representations"][33]
        emb = reps[0, 1:len(seq) + 1].mean(0).float().cpu().numpy()
        del results, tokens, reps
        torch.cuda.empty_cache()
        return emb

    n_done = n_failed = 0
    for i, uid in enumerate(need_extract):
        seq = store.uid_to_seq.get(uid)
        if seq is None:
            continue
        try:
            store.uid_to_esm2[uid] = extract_one(seq)
            n_done += 1
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] {uid} (len={len(seq)}) skipped")
            n_failed += 1
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [ERROR] {uid}: {type(e).__name__}: {e}")
            n_failed += 1
        if (i + 1) % checkpoint_every == 0:
            print(f"  {i + 1}/{len(need_extract)} (ok={n_done}, failed={n_failed})")
            np.save(esm2_path, store.uid_to_esm2)

    np.save(esm2_path, store.uid_to_esm2)
    print(f"ESM-2 extracted: {n_done:,} / failed: {n_failed:,} "
          f"-> total {len(store.uid_to_esm2):,}")


def build_xgb_features(df, store, use_protein="aac", use_ligand=True):
    features = []
    for _, row in df.iterrows():
        parts = []
        if use_protein == "aac":
            parts.append(store.uid_to_aac.get(row["uniprot_id"], np.zeros(20)))
        elif use_protein == "esm2":
            parts.append(store.uid_to_esm2.get(row["uniprot_id"], np.zeros(1280)))
        if use_ligand:
            parts.append(store.smiles_to_fp.get(row["smiles"], np.zeros(2048)))
        features.append(np.concatenate(parts) if parts else np.array([0.0]))
    return np.stack(features)
