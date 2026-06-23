"""Model architectures and per-seed training factories.

Models: XGBoost (protein / ligand / both / ESM features), DeepDTA, ESM2+MLP,
GraphDTA. Each factory returns `factory(train_df, val_df, seed) -> model_fn`,
where `model_fn(train_df, test_df) -> y_pred`. The last seed's weights are saved
for reuse (XGB-ESM is also kept in memory for SHAP).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor

from features import (
    AA_VOCAB,
    DEVICE,
    MAX_PROT_LEN,
    MAX_SMI_LEN,
    SMILES_VOCAB,
    build_xgb_features,
    encode_seq,
)


# ==========================================================================
# XGBoost
# ==========================================================================
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbosity": 0,
}


def make_xgb_factory(use_protein, use_ligand, model_name, store, seeds,
                     weight_dir, trained_xgb_models):
    def factory(train_df, val_df, seed):
        X_train = build_xgb_features(train_df, store, use_protein, use_ligand)
        model = XGBRegressor(**XGB_PARAMS, random_state=seed)
        model.fit(X_train, train_df["pKi"].values)

        if seed == seeds[-1]:
            trained_xgb_models[model_name] = (model, use_protein, use_ligand)
            model.save_model(weight_dir / f"{model_name}_seed{seed}.json")

        def model_fn(_train_unused, df_test):
            return model.predict(build_xgb_features(df_test, store, use_protein, use_ligand))

        return model_fn

    return factory


# ==========================================================================
# DeepDTA
# ==========================================================================
class DTADataset(Dataset):
    def __init__(self, df, store):
        self.data = df.reset_index(drop=True)
        self.store = store

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = self.store.uid_to_seq.get(row["uniprot_id"], "A")
        prot = torch.tensor(encode_seq(seq, AA_VOCAB, MAX_PROT_LEN), dtype=torch.long)
        lig = torch.tensor(encode_seq(row["smiles"], SMILES_VOCAB, MAX_SMI_LEN),
                           dtype=torch.long)
        return prot, lig, torch.tensor(row["pKi"], dtype=torch.float32)


class DeepDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = nn.Embedding(21, 128, padding_idx=0)
        self.le = nn.Embedding(65, 128, padding_idx=0)

        def conv_block(in_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, 32, 8), nn.ReLU(),
                nn.Conv1d(32, 64, 8), nn.ReLU(),
                nn.Conv1d(64, 96, 8), nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )

        self.pc = conv_block(128)
        self.lc = conv_block(128)
        self.fc = nn.Sequential(
            nn.Linear(192, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, p, l):
        p = self.pc(self.pe(p).permute(0, 2, 1)).squeeze(-1)
        l = self.lc(self.le(l).permute(0, 2, 1)).squeeze(-1)
        return self.fc(torch.cat([p, l], 1)).squeeze(-1)


def train_dl(model, train_df, val_df, store, epochs=50, bs=256, lr=1e-3, seed=42):
    """Train with ReduceLROnPlateau and keep the best-val checkpoint."""
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(DTADataset(train_df, store), bs, shuffle=True,
                    num_workers=12, pin_memory=True, generator=g)
    vl = DataLoader(DTADataset(val_df, store), bs, shuffle=False,
                    num_workers=12, pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss()
    best_loss, best_state = float("inf"), None

    for ep in range(epochs):
        model.train()
        for batch in tl:
            p, l, y = [b.to(DEVICE) for b in batch]
            opt.zero_grad()
            crit(model(p, l), y).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl_loss = np.mean([
                crit(model(*[b.to(DEVICE) for b in batch[:2]]),
                     batch[2].to(DEVICE)).item()
                for batch in vl])
        sch.step(vl_loss)

        if vl_loss < best_loss:
            best_loss = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"      epoch {ep + 1}/{epochs} | val_loss={vl_loss:.4f}")

    model.load_state_dict(best_state)
    return model, best_loss


def predict_dl(model, df_test, dataset_cls, store, bs=512):
    loader = DataLoader(dataset_cls(df_test, store), bs, shuffle=False, num_workers=12)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.extend(model(*[b.to(DEVICE) for b in batch[:-1]]).cpu().numpy())
    return np.array(preds)


def make_deepdta_factory(store, seeds, weight_dir):
    def factory(train_df, val_df, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = DeepDTA().to(DEVICE)
        model, best_loss = train_dl(model, train_df, val_df, store, seed=seed)
        print(f"    Best val loss: {best_loss:.4f}")
        if seed == seeds[-1]:
            torch.save(model.state_dict(), weight_dir / f"DeepDTA_seed{seed}.pt")
        return lambda _, df_test: predict_dl(model, df_test, DTADataset, store)

    return factory


# ==========================================================================
# ESM2 + MLP
# ==========================================================================
class ESM2Dataset(Dataset):
    def __init__(self, df, store):
        self.data = df.reset_index(drop=True)
        self.store = store

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pe = self.store.uid_to_esm2.get(row["uniprot_id"], np.zeros(1280))
        lf = self.store.smiles_to_fp.get(row["smiles"], np.zeros(2048))
        x = torch.tensor(np.concatenate([pe, lf]), dtype=torch.float32)
        return x, torch.tensor(row["pKi"], dtype=torch.float32)


class ESM2MLP(nn.Module):
    def __init__(self, dim=3328):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_esm2(model, train_df, val_df, store, epochs=50, bs=512, lr=1e-3, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(ESM2Dataset(train_df, store), bs, shuffle=True,
                    num_workers=12, pin_memory=True, generator=g)
    vl = DataLoader(ESM2Dataset(val_df, store), bs, shuffle=False,
                    num_workers=12, pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss()
    best_loss, best_state = float("inf"), None

    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl_loss = np.mean([crit(model(x.to(DEVICE)), y.to(DEVICE)).item()
                               for x, y in vl])
        sch.step(vl_loss)

        if vl_loss < best_loss:
            best_loss = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"      epoch {ep + 1}/{epochs} | val_loss={vl_loss:.4f}")

    model.load_state_dict(best_state)
    return model, best_loss


def predict_esm2(model, df_test, store, bs=1024):
    loader = DataLoader(ESM2Dataset(df_test, store), bs, shuffle=False, num_workers=12)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.extend(model(x.to(DEVICE)).cpu().numpy())
    return np.array(preds)


def make_esm2mlp_factory(store, seeds, weight_dir):
    def factory(train_df, val_df, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = ESM2MLP().to(DEVICE)
        model, best_loss = train_esm2(model, train_df, val_df, store, seed=seed)
        print(f"    Best val loss: {best_loss:.4f}")
        if seed == seeds[-1]:
            torch.save(model.state_dict(), weight_dir / f"ESM2MLP_seed{seed}.pt")
        return lambda _, df_test: predict_esm2(model, df_test, store)

    return factory


# ==========================================================================
# GraphDTA  (ligands that fail graph conversion are excluded, not imputed)
# ==========================================================================
class GraphDTADataset(Dataset):
    def __init__(self, df, store):
        self.data = df.reset_index(drop=True)
        self.store = store
        self.valid_idx = [i for i in range(len(self.data))
                          if self.data.iloc[i]["smiles"] in store.all_smiles_graphs]

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        row = self.data.iloc[self.valid_idx[idx]]
        seq = self.store.uid_to_seq.get(row["uniprot_id"], "A")
        prot = torch.tensor(encode_seq(seq, AA_VOCAB, MAX_PROT_LEN), dtype=torch.long)
        graph = self.store.all_smiles_graphs[row["smiles"]].clone()
        return {"prot": prot, "graph": graph,
                "label": torch.tensor(row["pKi"], dtype=torch.float32)}


def collate_graph(batch):
    from torch_geometric.data import Batch

    graph_batch = Batch.from_data_list([b["graph"] for b in batch])
    graph_batch.prot = torch.stack([b["prot"] for b in batch])
    return graph_batch, torch.stack([b["label"] for b in batch])


class GraphDTA(nn.Module):
    def __init__(self):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.pe = nn.Embedding(21, 128, padding_idx=0)
        self.pc = nn.Sequential(
            nn.Conv1d(128, 32, 8), nn.ReLU(),
            nn.Conv1d(32, 64, 8), nn.ReLU(),
            nn.Conv1d(64, 96, 8), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.gcn1 = GCNConv(5, 128)
        self.gcn2 = GCNConv(128, 128)
        self.gcn3 = GCNConv(128, 128)
        self.fc = nn.Sequential(
            nn.Linear(96 + 128, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool

        x = torch.relu(self.gcn1(data.x, data.edge_index))
        x = torch.relu(self.gcn2(x, data.edge_index))
        x = torch.relu(self.gcn3(x, data.edge_index))
        g = global_mean_pool(x, data.batch)
        p = self.pc(self.pe(data.prot).permute(0, 2, 1)).squeeze(-1)
        return self.fc(torch.cat([p, g], dim=1)).squeeze(-1)


def train_graphdta(train_df, val_df, store, epochs=50, bs=256, lr=1e-3, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(GraphDTADataset(train_df, store), bs, shuffle=True,
                    num_workers=12, collate_fn=collate_graph, generator=g)
    vl = DataLoader(GraphDTADataset(val_df, store), bs, shuffle=False,
                    num_workers=12, collate_fn=collate_graph)

    model = GraphDTA().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss()
    best_loss, best_state = float("inf"), None

    for ep in range(epochs):
        model.train()
        for graph_batch, labels in tl:
            graph_batch, labels = graph_batch.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            crit(model(graph_batch), labels).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl_loss = np.mean([
                crit(model(gb.to(DEVICE)), lab.to(DEVICE)).item()
                for gb, lab in vl])
        sch.step(vl_loss)

        if vl_loss < best_loss:
            best_loss = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"      epoch {ep + 1}/{epochs} | val_loss={vl_loss:.4f}")

    model.load_state_dict(best_state)
    return model, best_loss


def predict_graphdta(model, df_test, store, bs=256):
    """Predict only for convertible ligands; the rest stay NaN and are dropped
    downstream (a previous version imputed the mean, penalizing GraphDTA)."""
    ds = GraphDTADataset(df_test, store)
    full_preds = np.full(len(df_test), np.nan)
    if len(ds) == 0:
        return full_preds

    loader = DataLoader(ds, bs, shuffle=False, num_workers=12, collate_fn=collate_graph)
    model.eval()
    preds = []
    with torch.no_grad():
        for graph_batch, _ in loader:
            preds.extend(model(graph_batch.to(DEVICE)).cpu().numpy())
    full_preds[ds.valid_idx] = preds
    return full_preds


def make_graphdta_factory(store, seeds, weight_dir):
    def factory(train_df, val_df, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model, best_loss = train_graphdta(train_df, val_df, store, seed=seed)
        print(f"    Best val loss: {best_loss:.4f}")
        if seed == seeds[-1]:
            torch.save(model.state_dict(), weight_dir / f"GraphDTA_seed{seed}.pt")
        return lambda _, df_test: predict_graphdta(model, df_test, store)

    return factory
