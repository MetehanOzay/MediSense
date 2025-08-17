# eda.py — MediSense Cardio (BRFSS 2022) — ENGLISH VERSION
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Theme for better visuals
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# --------------------------------------------------
# Paths
ROOT = os.path.abspath(".")
CSV_PATH = os.path.join(ROOT, "data", "heart.csv")
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

print("Working directory:", ROOT)
print("CSV exists?:", os.path.exists(CSV_PATH))

# --------------------------------------------------
# Load data
df = pd.read_csv(CSV_PATH, low_memory=False)

# Target creation
if {"HadHeartAttack", "HadAngina"}.issubset(df.columns):
    df["HeartDisease"] = ((df["HadHeartAttack"] == "Yes") | (df["HadAngina"] == "Yes")).astype(int)
elif "HeartDisease" in df.columns:
    df["HeartDisease"] = df["HeartDisease"].astype(str).str.lower().map({"yes":1,"no":0})
else:
    raise ValueError("HeartDisease column not found.")

# --------------------------------------------------
# Basic info
print("Total records:", len(df))
print("HeartDisease distribution:\n", df["HeartDisease"].value_counts())

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved ->", path)

# --------------------------------------------------
# 1) Class distribution
plt.figure(figsize=(5,4))
sns.countplot(x="HeartDisease", data=df, hue="HeartDisease", legend=False, palette="Blues")
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0=No, 1=Yes)")
plt.ylabel("Count")
savefig(os.path.join(ASSETS, "class_dist.png"))

# --------------------------------------------------
# 2) Smoking status vs Heart Disease
smoke_col = "SmokerStatus" if "SmokerStatus" in df.columns else None
if smoke_col:
    plt.figure(figsize=(6,4))
    sns.countplot(x=smoke_col, hue="HeartDisease", data=df)
    plt.title("Smoking Status vs Heart Disease")
    plt.xlabel("Smoking Status")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    savefig(os.path.join(ASSETS, "smoking_vs_hd.png"))

# --------------------------------------------------
# 3) Alcohol consumption vs Heart Disease
alcohol_col = "AlcoholDrinkers" if "AlcoholDrinkers" in df.columns else None
if alcohol_col:
    plt.figure(figsize=(5,4))
    sns.countplot(x=alcohol_col, hue="HeartDisease", data=df)
    plt.title("Alcohol Consumption vs Heart Disease")
    plt.xlabel("Alcohol Consumption")
    plt.ylabel("Count")
    savefig(os.path.join(ASSETS, "alcohol_vs_hd.png"))

# --------------------------------------------------
# 4) Physical activity vs Heart Disease
pa_col = "PhysicalActivities" if "PhysicalActivities" in df.columns else None
if pa_col:
    plt.figure(figsize=(5,4))
    sns.countplot(x=pa_col, hue="HeartDisease", data=df)
    plt.title("Physical Activity vs Heart Disease")
    plt.xlabel("Physical Activity")
    plt.ylabel("Count")
    savefig(os.path.join(ASSETS, "physical_vs_hd.png"))

# --------------------------------------------------
# 5) BMI distribution
if "BMI" in df.columns:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df, x="BMI", hue="HeartDisease", common_norm=False, fill=True)
    plt.title("BMI Distribution by Heart Disease")
    plt.xlabel("BMI")
    plt.ylabel("Density")
    savefig(os.path.join(ASSETS, "bmi_kde.png"))

# --------------------------------------------------
# 6) Sleep hours distribution
sleep_col = "SleepHours" if "SleepHours" in df.columns else None
if sleep_col:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df, x=sleep_col, hue="HeartDisease", common_norm=False, fill=True)
    plt.title("Sleep Hours Distribution by Heart Disease")
    plt.xlabel("Sleep Hours")
    plt.ylabel("Density")
    savefig(os.path.join(ASSETS, "sleep_kde.png"))

# --------------------------------------------------
# 7) General health perception
gh_col = "GeneralHealth" if "GeneralHealth" in df.columns else None
if gh_col:
    plt.figure(figsize=(6,4))
    sns.countplot(x=gh_col, hue="HeartDisease", data=df,
                  order=["Excellent","Very good","Good","Fair","Poor"])
    plt.title("General Health vs Heart Disease")
    plt.xlabel("General Health")
    plt.ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    savefig(os.path.join(ASSETS, "genhealth_vs_hd.png"))

# --------------------------------------------------
# 8) List saved files
print("\nAssets folder content:")
for f in sorted(os.listdir(ASSETS)):
    print("-", f)
