"""LFW utility + CelebA gender-gap + vanishing-bias attribute filters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from genderbias.data.attrs import CelebAAttrDataset, resolve_celeba_root
from genderbias.data.pairs import build_calfw, build_cplfw, build_lfw
from genderbias.fairness.metrics import (
    best_threshold,
    cluster_correlated,
    gender_gap,
    pearson_corr_matrix,
    search_vanishing_filters,
)
from genderbias.models.fr_net import FaceEmbedder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer = 0.0, 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best_acc = max(best_acc, float((pred == labels).mean()))
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer


@torch.no_grad()
def eval_pairs(model, ds, device, batch_size, nw, tag) -> Dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    scores, labels = [], []
    model.eval()
    for a, b, y in tqdm(loader, desc=tag, leave=False):
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc, eer = best_acc_eer(s, y)
    print(f"[{tag}] acc={acc:.4f} eer={eer:.4f} n={len(y)}")
    return {"accuracy": acc, "eer": eer, "n_pairs": int(len(y))}


class CelebAProbePairs(Dataset):
    """Same/different identity pairs with gender + non-demographic attrs."""

    def __init__(self, base: CelebAAttrDataset, n_pos: int, n_neg: int, seed: int = 42):
        self.base = base
        rng = np.random.RandomState(seed)
        by_id: Dict[int, List[int]] = {}
        for i, (_fn, sid, _m, _nd) in enumerate(base.samples):
            by_id.setdefault(sid, []).append(i)
        ids = [k for k, v in by_id.items() if len(v) >= 2]
        pairs = []
        for _ in range(n_pos):
            sid = ids[rng.randint(len(ids))]
            i, j = rng.choice(by_id[sid], size=2, replace=False)
            pairs.append((int(i), int(j), 1))
        all_idx = list(range(len(base)))
        for _ in range(n_neg):
            i, j = rng.choice(all_idx, size=2, replace=False)
            while base.samples[i][1] == base.samples[j][1]:
                i, j = rng.choice(all_idx, size=2, replace=False)
            pairs.append((int(i), int(j), 0))
        rng.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, y = self.pairs[idx]
        ai, si, mi, ndi = self.base.samples[i]
        aj, sj, mj, ndj = self.base.samples[j]
        img_i = self.base.transform(Image.open(self.base.img_dir / ai).convert("RGB"))
        img_j = self.base.transform(Image.open(self.base.img_dir / aj).convert("RGB"))
        if mi == 0 and mj == 0:
            gp = 0
        elif mi == 1 and mj == 1:
            gp = 1
        else:
            gp = 2
        return img_i, img_j, y, gp, torch.from_numpy(ndi.astype(np.float32)), torch.from_numpy(
            ndj.astype(np.float32)
        )


@torch.no_grad()
def eval_gender_bias(model, probe: CelebAProbePairs, device, batch_size, nw, nd_names, min_pairs) -> Dict:
    loader = DataLoader(probe, batch_size=batch_size, shuffle=False, num_workers=nw)
    scores, labels, gps, attrs_a, attrs_b = [], [], [], [], []
    model.eval()
    for a, b, y, gp, nda, ndb in tqdm(loader, desc="celeba-gap", leave=False):
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
        gps.append(gp.numpy())
        attrs_a.append(nda.numpy())
        attrs_b.append(ndb.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    gp = np.concatenate(gps)
    aa = np.concatenate(attrs_a)
    ab = np.concatenate(attrs_b)
    thr = best_threshold(s, y)
    base = gender_gap(s, y, gp, thr)
    print(
        f"[gap] thr={thr:.3f} all={base['acc_all']:.4f} "
        f"FF={base['acc_ff']:.4f} MM={base['acc_mm']:.4f} gap={base['gap_mm_minus_ff']:.4f}"
    )
    # attribute correlation on unique images approximated via attr_a
    corr = pearson_corr_matrix(aa)
    clusters = cluster_correlated(nd_names, corr, thresh=0.5)
    filters = search_vanishing_filters(s, y, gp, aa, ab, nd_names, thr, min_pairs=min_pairs, top_k=12)
    best = filters[1] if len(filters) > 1 else None
    if best:
        print(
            f"[vanish] best={best['filter']} gap={best['gap']:.4f} "
            f"afi={best['afi']:.4f} n={best['n_pairs']}"
        )
    return {
        "threshold": thr,
        "baseline": base,
        "n_attr_clusters": len(clusters),
        "attr_clusters_sample": clusters[:8],
        "vanishing_filters": filters,
    }


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    emb_dim = int(blob.get("emb_dim", cfg["model"].get("emb_dim", 128)))
    model = FaceEmbedder(int(blob["num_classes"]), emb_dim=emb_dim, pretrained=False).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    image_size = int(cfg["model"].get("image_size", 112))
    bs = int(cfg["eval"].get("batch_size", 128))
    nw = int(cfg["train"].get("num_workers", 4))
    max_p = cfg["eval"].get("max_lfw_pairs")
    max_p = int(max_p) if max_p is not None else None

    results: Dict = {"verification": {}, "fairness": {}}
    for name, builder in (("lfw", build_lfw), ("calfw", build_calfw), ("cplfw", build_cplfw)):
        try:
            ds = builder(cfg, image_size, max_p)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        results["verification"][name] = eval_pairs(model, ds, device, bs, nw, name)

    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    val_ds = CelebAAttrDataset(
        root,
        split="val",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=int(cfg["train"].get("seed", 42)),
        train=False,
    )
    n_pos = int(cfg["eval"].get("n_pos_pairs", 2000))
    n_neg = int(cfg["eval"].get("n_neg_pairs", 2000))
    # smoke: shrink if val small
    n_pos = min(n_pos, max(10, len(val_ds) // 4))
    n_neg = min(n_neg, max(10, len(val_ds) // 4))
    probe = CelebAProbePairs(val_ds, n_pos=n_pos, n_neg=n_neg, seed=int(cfg["train"].get("seed", 42)))
    nd_names = blob.get("nd_names") or val_ds.nd_names
    min_pairs = int(cfg["eval"].get("min_filter_pairs", 40))
    results["fairness"] = eval_gender_bias(
        model, probe, device, min(bs, 64), nw, nd_names, min_pairs
    )

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
