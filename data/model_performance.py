# data/model_performance.py
# RandomForest performans metrikleri + ROC/CM görselleri + KMeans (k=3) cluster özeti
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    silhouette_score, adjusted_rand_score
)
from sklearn.cluster import KMeans

# ================== YOLLAR (mutlak) ==================
BASE_DIR = Path(__file__).resolve().parent            # .../medisense/data
CSV_PATH = BASE_DIR / "heart.csv"                    # .../medisense/data/heart.csv
OUT_DIR  = BASE_DIR.parent / "assets"                # .../medisense/assets
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("OUT_DIR ->", OUT_DIR)

def die(msg):
    print(f"❌ {msg}")
    sys.exit(1)

def main():
    print("✅ Script başladı…")

    # ============== 1) Veri ==============
    if not CSV_PATH.exists():
        die(f"CSV bulunamadı: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"✅ Veri yüklendi. Şekil: {df.shape}")

    if "HadHeartAttack" not in df.columns:
        die("Hedef kolon 'HadHeartAttack' bulunamadı.")

    # Hedefi 0/1'e çevir
    y = df["HadHeartAttack"]
    if y.dtype == object:
        y = y.map({"Yes": 1, "No": 0})
    y = pd.to_numeric(y, errors="coerce")
    if y.isna().all():
        die("Hedef 0/1'e çevrilemedi (tümü NaN).")

    # Özellikler (hedef hariç)
    X = df.drop(columns=["HadHeartAttack"])

    # Sayısal/kategorik ayrım
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    print(f"🔧 Sayısal: {len(num_cols)} | Kategorik: {len(cat_cols)}")

    # ============== 2) Pipeline (OneHot + RF) ==============
    preprocess = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    pipe = Pipeline([("preprocess", preprocess), ("clf", rf)])

    # Train/Test ayrımı (stratify: sınıf dengesini koru)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int), test_size=0.20, random_state=42, stratify=y
    )
    print(f"✅ Train/Test: train={X_train.shape[0]}, test={X_test.shape[0]}")

    # Fit
    pipe.fit(X_train, y_train)

    # ============== 3) RF Performans ==============
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    print("\n=== RANDOM FOREST — Test Sonuçları ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-Score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

    # Confusion Matrix (PNG)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix — RandomForest")
    plt.colorbar()
    plt.xticks([0,1], ["No", "Yes"])
    plt.yticks([0,1], ["No", "Yes"])
    for (i,j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.tight_layout()
    cm_path = OUT_DIR / "rf_confusion.png"
    plt.savefig(str(cm_path), dpi=220, bbox_inches="tight"); plt.close()

    # ROC Curve (PNG)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(4,3))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1], [0,1], "--", alpha=0.6)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — RandomForest")
    plt.legend()
    plt.tight_layout()
    roc_path = OUT_DIR / "rf_roc.png"
    plt.savefig(str(roc_path), dpi=220, bbox_inches="tight"); plt.close()

    # CSV olarak özet metrikler (sunum için)
    perf_csv = OUT_DIR / "rf_performance_summary.csv"
    pd.DataFrame([{
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3),
        "ROC_AUC": round(auc, 3)
    }]).to_csv(str(perf_csv), index=False)

    print(f"🖼️ Kaydedildi: {cm_path}")
    print(f"🖼️ Kaydedildi: {roc_path}")
    print(f"🗂️ Kaydedildi: {perf_csv}")

    # ============== 4) K-Means (k=3) — keşifsel ==============
    print("\n=== K-MEANS — Keşifsel Kümeleme (k=3) ===")
    # Tüm veriyi aynı preprocess ile dönüştürüp KMeans uygula
    X_all_trans = pipe.named_steps["preprocess"].fit_transform(X)
    km = KMeans(n_clusters=3, random_state=42, n_init="auto")
    clusters = km.fit_predict(X_all_trans)

    sil = silhouette_score(X_all_trans, clusters)
    ari = adjusted_rand_score(y, clusters)  # gerçek etiketle uyum (0-1)
    print(f"Silhouette Score    : {sil:.3f}  (küme ayrışma kalitesi)")
    print(f"Adjusted Rand Index : {ari:.3f}  (gerçek etiket uyumu)")

    # Küme bazında kalp krizi oranı
    summary = []
    for c in range(3):
        idx = (clusters == c)
        n = int(idx.sum())
        rate = float(y[idx].mean()) if n > 0 else 0.0
        summary.append({"cluster": c, "count": n, "RatePct": round(rate*100, 2)})

    sum_df = pd.DataFrame(summary).sort_values("RatePct").reset_index(drop=True)
    sum_df["risk_level"] = ["Low", "Medium", "High"][:len(sum_df)]
    csv_out = OUT_DIR / "kmeans3_summary.csv"
    sum_df.to_csv(str(csv_out), index=False)
    print("Cluster summary:\n", sum_df)
    print(f"🗂️ Kaydedildi: {csv_out}")

    print("\n✅ Bitti.")

if __name__ == "__main__":
    main()
