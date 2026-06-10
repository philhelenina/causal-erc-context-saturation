#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train lexical-only classifier (SenticNet 4D axes) for IEMOCAP (4-way or 6-way).
"""
import os

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DATA_DIRS = {
    "4way": HOME / "data" / "iemocap_4way_data",
    "6way": HOME / "data" / "iemocap_6way_data",
}
LABEL_MAPS = {
    "4way": {"ang":0,"hap":1,"sad":2,"neu":3},
    "6way": {"ang":0,"hap":1,"sad":2,"neu":3,"exc":4,"fru":5}
}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_split(root: Path, split: str):
    npz = np.load(root/f"{split}.npz")
    X = npz["X"] if "X" in npz else npz["embeddings"]
    y = npz["labels"]
    mask = y >= 0
    return X[mask], y[mask]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval(); preds=[]; gts=[]
    for xb,yb in loader:
        xb,yb=xb.to(device), yb.to(device)
        pred=torch.argmax(model(xb),dim=-1)
        preds.append(pred.cpu().numpy()); gts.append(yb.cpu().numpy())
    y_pred=np.concatenate(preds); y_true=np.concatenate(gts)
    return dict(
        acc=accuracy_score(y_true,y_pred),
        f1m=f1_score(y_true,y_pred,average="macro",zero_division=0),
        f1w=f1_score(y_true,y_pred,average="weighted",zero_division=0)
    )

def train_loop(model, loaders, device, epochs, lr, wd, patience):
    tr_loader,va_loader,te_loader=loaders
    crit=nn.CrossEntropyLoss()
    optim=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    best={"f1m":-1,"state":None}; wait=0
    for ep in range(1,epochs+1):
        model.train()
        for xb,yb in tr_loader:
            xb,yb=xb.to(device).float(), yb.to(device).long()
            loss=crit(model(xb),yb)
            optim.zero_grad(); loss.backward(); optim.step()
        va=eval_model(model,va_loader,device)
        if va["f1m"]>best["f1m"]:
            best.update(f1m=va["f1m"],state=model.state_dict()); wait=0
        else: wait+=1
        if wait>=patience: break
    if best["state"] is not None: model.load_state_dict(best["state"])
    return eval_model(model,te_loader,device)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", choices=["4way","6way"], required=True)
    ap.add_argument("--lex_root", type=str, required=True,
                    help="Path to senticnet-axes directory containing train/val/test.npz")
    ap.add_argument("--model", choices=["mlp"], default="mlp")
    ap.add_argument("--hidden_size", type=int, default=192)
    ap.add_argument("--dropout_rate", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_epochs", type=int, default=200)
    ap.add_argument("--early_stopping_patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, required=True)
    args=ap.parse_args()

    set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root=Path(args.lex_root)
    Xtr,ytr=load_split(root,"train")
    Xva,yva=load_split(root,"val")
    Xte,yte=load_split(root,"test")

    loaders=(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),
                                         batch_size=args.batch_size,shuffle=True),
             torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xva),torch.from_numpy(yva)),
                                         batch_size=args.batch_size),
             torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(Xte),torch.from_numpy(yte)),
                                         batch_size=args.batch_size))

    model=MLP(Xtr.shape[1], args.hidden_size, num_classes=len(LABEL_MAPS[args.task]), dropout=args.dropout_rate).to(device)
    results=train_loop(model,loaders,device,args.num_epochs,args.learning_rate,args.weight_decay,args.early_stopping_patience)

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    with open(out/"results.json","w") as f: json.dump(results,f,indent=2)
    print(f"[OK] {args.task} Results → {out}/results.json :: {results}")

if __name__=="__main__":
    main()
