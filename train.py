
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib, traceback

ROOT = os.path.abspath(".")
CSV_PATH = os.path.join(ROOT, "data", "heart.csv")
ART_DIR  = os.path.join(ROOT, "artifacts")
ASSETS   = os.path.join(ROOT, "assets")
os.makedirs(ART_DIR, exist_ok=True); os.makedirs(ASSETS, exist_ok=True)

print("Working dir:", ROOT)
print("CSV exists? ", os.path.exists(CSV_PATH))
df = pd.read_csv(CSV_PATH, low_memory=False)


if {"HadHeartAttack","HadAngina"}.issubset(df.columns):
    df["HeartDisease"] = ((df["HadHeartAttack"]=="Yes") | (df["HadAngina"]=="Yes")).astype(int)
elif "HeartDisease" in df.columns:
    df["HeartDisease"] = df["HeartDisease"].astype(str).str.lower().map({"yes":1,"no":0}).astype(int)
else:
    raise ValueError("Heart disease info not found.")
print("Target distribution:\n", df["HeartDisease"].value_counts())

# ---- 2) Kullanılacak kolonlar (bu veri setine göre)
INPUT_COLS = [
    "BMI","PhysicalActivities","SleepHours","GeneralHealth",
    "HadDiabetes","AgeCategory","Sex","AlcoholDrinkers","SmokerStatus"
]
missing = [c for c in INPUT_COLS if c not in df.columns]
if missing: raise ValueError(f"Missing columns: {missing}")

X = df[INPUT_COLS].copy()
y = df["HeartDisease"].astype(int)


num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


lr_pipe = Pipeline([("preprocess", preprocess),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
lr_pipe.fit(Xtr, ytr)
lr_proba = lr_pipe.predict_proba(Xte)[:,1]
lr_pred  = (lr_proba >= 0.5).astype(int)
lr_acc, lr_f1, lr_auc = accuracy_score(yte, lr_pred), f1_score(yte, lr_pred), roc_auc_score(yte, lr_proba)
print(f"LogisticRegression -> Acc: {lr_acc:.3f} | F1: {lr_f1:.3f} | AUC: {lr_auc:.3f}")

# ---- 6) RANDOM FOREST
try:
    tr_idx = Xtr.sample(frac=0.30, random_state=42).index
    Xtr_rf, ytr_rf = Xtr.loc[tr_idx], ytr.loc[tr_idx]

    rf_pipe = Pipeline([("preprocess", preprocess),
                        ("model", RandomForestClassifier(
                            n_estimators=200, max_depth=15, random_state=42,
                            class_weight="balanced_subsample", n_jobs=-1
                        ))])
    rf_pipe.fit(Xtr_rf, ytr_rf)
    rf_proba = rf_pipe.predict_proba(Xte)[:,1]
    rf_pred  = (rf_proba >= 0.5).astype(int)
    rf_acc, rf_f1, rf_auc = accuracy_score(yte, rf_pred), f1_score(yte, rf_pred), roc_auc_score(yte, rf_proba)
    print(f"RandomForest       -> Acc: {rf_acc:.3f} | F1: {rf_f1:.3f} | AUC: {rf_auc:.3f}")
except Exception as e:
    print("[WARN] RandomForest skipped:", e)
    traceback.print_exc()
    rf_pipe = None; rf_auc = -1; rf_pred = None; rf_proba = None


best_name, best_auc, best_pipe, best_pred, best_proba = ("LR", lr_auc, lr_pipe, lr_pred, lr_proba)
if rf_pipe is not None and rf_auc > best_auc:
    best_name, best_auc, best_pipe, best_pred, best_proba = ("RF", rf_auc, rf_pipe, rf_pred, rf_proba)

joblib.dump(best_pipe, os.path.join(ART_DIR, "model.joblib"))
joblib.dump(INPUT_COLS, os.path.join(ART_DIR, "feature_order.joblib"))
print(f"Saved best model: {best_name} (AUC={best_auc:.3f}) -> artifacts/model.joblib")

# ---- 8) Confusion & ROC (best model)
sns.set_theme(style="whitegrid", palette="Set2")
cm = confusion_matrix(yte, best_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig(os.path.join(ASSETS, "confusion.png"), dpi=150); plt.close()
print("Saved -> assets/confusion.png")

plt.figure()
RocCurveDisplay.from_predictions(yte, best_proba)
plt.title(f"ROC Curve — {best_name}")
plt.tight_layout(); plt.savefig(os.path.join(ASSETS, "roc.png"), dpi=150); plt.close()
print("Saved -> assets/roc.png")

# ---- 9) Model comparison tablosu
pd.DataFrame(
    [["LogisticRegression", lr_acc, lr_f1, lr_auc],
     ["RandomForest",       (rf_acc if rf_pipe else None),
                           (rf_f1 if rf_pipe else None),
                           (rf_auc if rf_pipe else None)]],
    columns=["Model","Accuracy","F1","ROC_AUC"]
).to_csv(os.path.join(ASSETS, "model_comparison.csv"), index=False)
print("Saved -> assets/model_comparison.csv")

try:

    sample_size = min(len(X), 30000)
    Xs = X.sample(n=sample_size, random_state=42)
    ys = y.loc[Xs.index]

    Z = preprocess.fit_transform(Xs)
    Zd = Z.toarray() if hasattr(Z, "toarray") else Z

    pca = PCA(n_components=2, random_state=42)
    Z2 = pca.fit_transform(Zd)
    expl = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={expl[0]:.2f}, PC2={expl[1]:.2f}")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(Z2)

    plt.figure(figsize=(6,5))
    for c in range(3):
        idx = clusters == c
        plt.scatter(Z2[idx,0], Z2[idx,1], s=8, label=f"Cluster {c}")
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=120, c="black", marker="X", label="Centroids")
    plt.title("PCA (2D) with K-Means Clusters")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(ASSETS, "pca_kmeans.png"), dpi=150); plt.close()
    print("Saved -> assets/pca_kmeans.png")

    pd.crosstab(pd.Series(clusters, name="Cluster"), pd.Series(ys.values, name="HeartDisease")).to_csv(
        os.path.join(ASSETS, "clusters_vs_target.csv"))
    print("Saved -> assets/clusters_vs_target.csv")
except Exception as e:
    print("[WARN] PCA/KMeans skipped:", e)
    traceback.print_exc()

print("All done.")

