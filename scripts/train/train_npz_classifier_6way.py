#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_npz_classifier_6way.py (VERBOSE VERSION)
- IEMOCAP 6-way 전용
- 실시간 성능 출력 추가
"""
import os

import argparse, json, math, random, sys
from pathlib import Path
from typing import Optional
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
DEFAULT_EMB_ROOT = HOME / "data" / "embeddings" / "6way" / "sentence-roberta"
CSV_BASE = HOME / "data" / "iemocap_6way_data"
CSV_NAME = "{split}_6way_unified.csv"

LABEL_TO_ID = {"ang":0,"hap":1,"sad":2,"neu":3,"exc":4,"fru":5}
ID_TO_LABEL = {v:k for k,v in LABEL_TO_ID.items()}
NUM_CLASSES = 6

# ---------------- Utils ----------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_npz_embeddings(emb_root: Path, layer: str, pool: str, split: str) -> np.ndarray:
    p = emb_root / layer / pool / f"{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"❌ Embedding file not found: {p}")
    arr = np.load(p)
    emb = (arr["embeddings"] if "embeddings" in arr else arr[list(arr.keys())[0]]).astype(np.float32)
    return emb

def load_labels_with_mask(split: str, label_col_pref: Optional[str] = None):
    csv_path = CSV_BASE / CSV_NAME.format(split=split)
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)

    y = None
    if label_col_pref and label_col_pref in df.columns:
        y = pd.to_numeric(df[label_col_pref], errors="coerce")
    elif "label_num" in df.columns:
        y = pd.to_numeric(df["label_num"], errors="coerce")
    else:
        lab = df["label"].astype(str).str.strip().str.lower()
        y = lab.map(LABEL_TO_ID)

    y = y.fillna(-1).astype(int)
    mask = (y >= 0)
    return y.to_numpy(), mask.to_numpy()

def align_xy(X,y,mask,name):
    if len(X)!=len(mask):
        n=min(len(X),len(mask)); X,y,mask=X[:n],y[:n],mask[:n]
    X_filtered, y_filtered = X[mask], y[mask]
    return X_filtered, y_filtered

# ---------------- Models ----------------
class MLP(nn.Module):
    def __init__(self,in_dim,hidden,num_classes,drop):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),
                               nn.Dropout(drop),nn.Linear(hidden,num_classes))
    def forward(self,x):
        if x.dim()==3: x=x.reshape(x.size(0),-1)
        return self.net(x)

class LSTMClassifier(nn.Module):
    def __init__(self,in_dim,hidden,num_classes,drop):
        super().__init__()
        self.lstm=nn.LSTM(in_dim,hidden,1,batch_first=True)
        self.proj=nn.Sequential(nn.Dropout(drop),nn.Linear(hidden,num_classes))
    def forward(self,x):
        if x.dim()==2: x=x.unsqueeze(1)
        out,_=self.lstm(x)
        return self.proj(out[:,-1,:])

# ---------------- Train/Eval ----------------
def compute_accuracy(model, dataloader, device):
    """Compute accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device).float(), yb.to(device).long()
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0

def train_one(args,Xtr,ytr,Xva,yva,in_dim:int):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    model = MLP(in_dim,args.hidden_size,NUM_CLASSES,args.dropout_rate) if args.model=="mlp" \
            else LSTMClassifier(in_dim,args.hidden_size,NUM_CLASSES,args.dropout_rate)
    model.to(device)
    
    print(f"\n{'='*60}")
    print(f"Model: {args.model.upper()} | Hidden: {args.hidden_size} | Dropout: {args.dropout_rate}")
    print(f"LR: {args.learning_rate} | WD: {args.weight_decay} | BS: {args.batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    crit=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    tr_dl=torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),
        batch_size=args.batch_size,shuffle=True)
    va_dl=torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xva),torch.from_numpy(yva)),
        batch_size=args.batch_size)
    
    best,best_state,wait=1e9,None,0
    best_epoch = 0
    
    print(f"{'Epoch':<6} {'TrLoss':<10} {'ValLoss':<10} {'ValAcc':<10} {'Status':<15}")
    print(f"{'-'*60}")
    
    for ep in range(1,args.num_epochs+1):
        # Training
        model.train()
        train_loss = 0.0
        for xb,yb in tr_dl:
            xb,yb=xb.to(device).float(),yb.to(device).long()
            opt.zero_grad()
            loss=crit(model(xb),yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(tr_dl.dataset)
        
        # Validation
        va_loss=0
        model.eval()
        with torch.no_grad():
            for xb,yb in va_dl:
                xb,yb=xb.to(device).float(),yb.to(device).long()
                va_loss+=crit(model(xb),yb).item()*xb.size(0)
        va_loss/=len(va_dl.dataset)
        
        # Validation accuracy
        val_acc = compute_accuracy(model, va_dl, device)
        
        # Early stopping check
        status = ""
        if va_loss<best-1e-6:
            best=va_loss
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            wait=0
            best_epoch = ep
            status = "✓ BEST"
        else:
            wait+=1
            if wait>=args.early_stopping_patience:
                status = "✗ EARLY STOP"
            else:
                status = f"wait {wait}/{args.early_stopping_patience}"
        
        # Print every 10 epochs or if best/stopped
        if ep % 10 == 0 or "BEST" in status or "STOP" in status:
            print(f"{ep:<6} {train_loss:<10.4f} {va_loss:<10.4f} {val_acc:<10.4f} {status:<15}")
            sys.stdout.flush()
        
        if wait>=args.early_stopping_patience:
            print(f"\nEarly stopped at epoch {ep}. Best was epoch {best_epoch}.\n")
            break
    
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n✅ Loaded best model from epoch {best_epoch} (val_loss: {best:.4f})\n")
    
    return model

def eval_and_save(args,model,Xte,yte,out_dir,embed_name):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds=[]
    te_dl=torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xte),torch.from_numpy(yte)),
        batch_size=args.batch_size)
    
    with torch.no_grad():
        for xb,_ in te_dl:
            xb=xb.to(device).float()
            preds.append(torch.argmax(model(xb),dim=1).cpu().numpy())
    
    yhat=np.concatenate(preds)
    acc=accuracy_score(yte,yhat)
    f1m=f1_score(yte,yhat,average="macro")
    f1w=f1_score(yte,yhat,average="weighted")
    cm=confusion_matrix(yte,yhat,labels=list(range(NUM_CLASSES)))
    
    # Per-class F1 scores
    f1_per_class = f1_score(yte, yhat, average=None)
    
    # Classification report
    class_report = classification_report(
        yte, yhat, 
        target_names=[ID_TO_LABEL[i] for i in range(NUM_CLASSES)],
        digits=4
    )
    
    # Print results to stdout
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Macro F1:     {f1m:.4f}")
    print(f"Weighted F1:  {f1w:.4f}")
    print(f"\nPer-class F1:")
    for i, label in enumerate([ID_TO_LABEL[j] for j in range(NUM_CLASSES)]):
        print(f"  {label}: {f1_per_class[i]:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nDetailed Classification Report:")
    print(class_report)
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    # Save to JSON
    out_dir.mkdir(parents=True,exist_ok=True)
    results = {
        "metrics": {
            "accuracy": float(acc),
            "macro_f1": float(f1m),
            "weighted_f1": float(f1w)
        },
        "per_class_f1": {
            ID_TO_LABEL[i]: float(f1_per_class[i]) 
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "config": {
            "model": args.model,
            "layer": args.layer,
            "pool": args.pool,
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "dropout_rate": args.dropout_rate,
            "seed": args.seed
        }
    }
    
    with open(out_dir/"results.json","w") as f:
        json.dump(results, f, indent=2)
    
    return acc, f1m, f1w

# ---------------- Main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--layer",required=True)
    ap.add_argument("--pool",required=True)
    ap.add_argument("--model",choices=["mlp","lstm"],default="mlp")
    ap.add_argument("--hidden_size",type=int,default=256)
    ap.add_argument("--batch_size",type=int,default=64)
    ap.add_argument("--learning_rate",type=float,default=1e-3)
    ap.add_argument("--weight_decay",type=float,default=0.0)
    ap.add_argument("--num_epochs",type=int,default=200)
    ap.add_argument("--early_stopping_patience",type=int,default=60)
    ap.add_argument("--dropout_rate",type=float,default=0.3)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--label_col",type=str,default=None)
    ap.add_argument("--out_dir",type=str,required=True)
    ap.add_argument("--emb_root",type=str,default="",help="override embeddings root")
    args=ap.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"🚀 Starting 6-way Classification")
    print(f"{'#'*60}")
    print(f"Config: {args.layer}/{args.pool}/{args.model}/seed_{args.seed}")
    print(f"{'#'*60}\n")
    
    set_seed(args.seed)

    # Load embeddings
    emb_root=Path(args.emb_root) if args.emb_root else DEFAULT_EMB_ROOT
    print("📂 Loading embeddings...")
    try:
        Xtr=load_npz_embeddings(emb_root,args.layer,args.pool,"train")
        Xva=load_npz_embeddings(emb_root,args.layer,args.pool,"val")
        Xte=load_npz_embeddings(emb_root,args.layer,args.pool,"test")
        print(f"  ✓ Train: {Xtr.shape}")
        print(f"  ✓ Val:   {Xva.shape}")
        print(f"  ✓ Test:  {Xte.shape}")
    except Exception as e:
        print(f"\n❌ ERROR loading embeddings: {e}\n")
        sys.exit(1)
    
    # Load labels
    print("\n🏷️  Loading labels...")
    try:
        ytr,mtr=load_labels_with_mask("train",args.label_col)
        yva,mva=load_labels_with_mask("val",args.label_col)
        yte,mte=load_labels_with_mask("test",args.label_col)
        print(f"  ✓ Train: {len(ytr)} labels, {mtr.sum()} valid")
        print(f"  ✓ Val:   {len(yva)} labels, {mva.sum()} valid")
        print(f"  ✓ Test:  {len(yte)} labels, {mte.sum()} valid")
    except Exception as e:
        print(f"\n❌ ERROR loading labels: {e}\n")
        sys.exit(1)
    
    # Align data
    print("\n🔗 Aligning data...")
    Xtr,ytr=align_xy(Xtr,ytr,mtr,"train")
    Xva,yva=align_xy(Xva,yva,mva,"val")
    Xte,yte=align_xy(Xte,yte,mte,"test")
    
    print(f"  After filtering:")
    print(f"    Train: {Xtr.shape[0]} samples")
    print(f"    Val:   {Xva.shape[0]} samples")
    print(f"    Test:  {Xte.shape[0]} samples")
    
    # Check label distribution
    print(f"\n📊 Label distribution:")
    for split_name, y_split in [("Train", ytr), ("Val", yva), ("Test", yte)]:
        counts = np.bincount(y_split, minlength=NUM_CLASSES)
        print(f"  {split_name}:", end=" ")
        for i, label in enumerate([ID_TO_LABEL[j] for j in range(NUM_CLASSES)]):
            print(f"{label}:{counts[i]}", end=" ")
        print()
    
    # Check for issues
    if Xtr.shape[0] == 0 or Xva.shape[0] == 0 or Xte.shape[0] == 0:
        print("\n❌ ERROR: One or more splits have 0 samples after filtering!\n")
        sys.exit(1)
    
    # Check if any class is missing
    for split_name, y_split in [("Train", ytr), ("Val", yva), ("Test", yte)]:
        unique_labels = np.unique(y_split)
        if len(unique_labels) < NUM_CLASSES:
            print(f"\n⚠️  WARNING: {split_name} split is missing some classes!")
            print(f"   Expected: {NUM_CLASSES} classes, Got: {len(unique_labels)} classes")
    
    in_dim=Xtr.shape[-1]
    print(f"\n🧠 Input dimension: {in_dim}")
    
    # Train
    print(f"\n{'='*60}")
    print(f"🏋️  Training...")
    print(f"{'='*60}")
    model=train_one(args,Xtr,ytr,Xva,yva,in_dim)
    
    # Evaluate
    print(f"\n{'='*60}")
    print(f"🧪 Evaluating on test set...")
    print(f"{'='*60}")
    eval_and_save(args,model,Xte,yte,Path(args.out_dir),f"sr6_{args.layer}_{args.pool}")
    
    print(f"\n✅ Done! Results saved to: {args.out_dir}\n")

if __name__=="__main__": main()