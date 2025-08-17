# pca_kmeans.py — PCA(2D) + KMeans(k=3) with risk labeling
import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

CSV_PATH   = "data/heart.csv"
MODEL_PATH = "artifacts/model.joblib"          # sadece preprocess'i almak için, olmazsa fallback
FEAT_PATH  = "artifacts/feature_order.joblib"
OUT_DIR    = "assets"
os.makedirs(OUT_DIR, exist_ok=True)

def to_dense(X):
    if hasattr(X, "toarray"): return X.toarray()
    if hasattr(X, "values"):  return X.values
    return np.asarray(X)

# 1) Veri
df = pd.read_csv(CSV_PATH, low_memory=False)
if "HadHeartAttack" not in df.columns:
    sys.exit("❌ 'HadHeartAttack' column is required in CSV.")
y = df["HadHeartAttack"]
if y.dtype == object:
    y = y.map({"Yes": 1, "No": 0})
y = pd.to_numeric(y, errors="coerce")

# 2) Özellikler
feature_order = joblib.load(FEAT_PATH)
X_raw = df[feature_order].copy()

# 3) Satır temizliği (X ve y eksiksiz)
mask = X_raw.notna().all(axis=1) & (~y.isna())
X_raw = X_raw.loc[mask]
y = y.loc[mask].astype(int)

# 4) Eğitimdeki preprocess'i kullan (varsa)
Xt = None
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

if isinstance(model, Pipeline) and len(model.steps) >= 2:
    preprocess = Pipeline(model.steps[:-1])      # son adım sınıflandırıcı
    Xt = preprocess.transform(X_raw)
else:
    # named_steps ile arama
    preprocess = getattr(getattr(model, "named_steps", {}), "get", lambda *_: None)("preprocess") if model else None
    if preprocess is not None:
        Xt = preprocess.transform(X_raw)
    else:
        # fallback: sadece görselleştirme için dummies
        Xt = pd.get_dummies(X_raw, drop_first=True).fillna(0)

Xt = to_dense(Xt)
if Xt.ndim != 2 or Xt.shape[0] == 0:
    sys.exit(f"❌ Transform error. Xt shape={getattr(Xt,'shape',None)}")

print(f"✅ Data ready: X={Xt.shape}, y={y.shape}")

# 5) PCA(2)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(Xt)
pc1, pc2 = pca.explained_variance_ratio_[:2]
print(f"✅ PCA explained variance: PC1={pc1:.2f}, PC2={pc2:.2f}")

# 6) KMeans(k=3)
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = km.fit_predict(X_pca)
centers = km.cluster_centers_

# 7) Her kümenin risk oranı (gerçek etiket)
summary = []
for c in range(k):
    idx = (labels == c)
    n = idx.sum()
    pos_rate = float(y[idx].mean()) if n > 0 else 0.0   # 0..1
    summary.append({"cluster": c, "size": n, "heart_attack_rate": pos_rate})

sum_df = pd.DataFrame(summary).sort_values("heart_attack_rate").reset_index(drop=True)

# Low/Medium/High etiketlerini atayalım (düşükten yükseğe sıraya göre)
tiers = ["Low", "Medium", "High"]
sum_df["tier"] = tiers[:len(sum_df)]
# Orijinal cluster id -> tier eşlemesi
cluster_to_tier = {row["cluster"]: row["tier"] for _, row in sum_df.iterrows()}

# 8) Görsel
colors = {"Low":"#38c172", "Medium":"#ffd166", "High":"#ef476f"}
cluster_colors = np.array([colors[cluster_to_tier[c]] for c in labels])

plt.figure(figsize=(10,8))
plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster_colors, s=8, alpha=0.75, edgecolors="none")
# centroid ve etiket
for c in range(k):
    cx, cy = centers[c]
    tier = cluster_to_tier[c]
    rate = next(r for r in summary if r["cluster"]==c)["heart_attack_rate"] * 100
    plt.scatter([cx],[cy], s=220, c="black", marker="X")
    plt.text(cx, cy, f"{tier}\n{rate:.1f}%", ha="center", va="center", color="white",
             bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.7))

plt.title(f"PCA 2D + KMeans (k=3) — Clusters labeled by true heart-attack rate\nPC1={pc1:.2f}, PC2={pc2:.2f}")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, alpha=0.25)
out_png = os.path.join(OUT_DIR, "pca_kmeans3.png")
plt.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close()

# 9) Özet tabloyu kaydet (sunumda kullanışlı)
sum_df["rate_pct"] = (sum_df["heart_attack_rate"] * 100).round(2)
sum_df = sum_df[["tier","cluster","size","rate_pct"]]
sum_path = os.path.join(OUT_DIR, "kmeans3_summary.csv")
sum_df.to_csv(sum_path, index=False)

print(f"✅ Saved figure: {out_png}")
print(f"✅ Saved summary: {sum_path}")
print(sum_df)


