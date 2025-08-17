# app.py — MediSense (BRFSS 2022) — modular UI + tips + foods gallery + % map
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import json
import os
import plotly.graph_objects as go  # ADD

# ==== Helpers for explanation & report ====
import datetime as _dt

print("✅ Script başladı...")

# Risk-tier renk rozeti
def tier_badge(tier: str):
    return "🟥 High" if tier=="High" else ("🟧 Medium" if tier=="Medium" else "🟩 Low")

# Modeli tek satır sözlükle çalıştırıp olasılık döndür
def _predict_prob_from_dict(model, feature_order, feat_dict: dict) -> float:
    import pandas as _pd
    row = _pd.DataFrame([feat_dict], columns=feature_order)
    return float(model.predict_proba(row)[:, 1])

# Kullanıcı girdisini, session_state’te sakladığımız profile’dan baz alarak hazırla
def _current_features_for_model():
    prof = st.session_state.get("profile", {})
    return {
        "BMI": prof.get("BMI", 25),
        "PhysicalActivities": prof.get("PhysicalActivities", "Yes") if "PhysicalActivities" in prof else "Yes",
        "SleepHours": prof.get("SleepHours", 7.0) if "SleepHours" in prof else 7.0,
        "GeneralHealth": prof.get("GeneralHealth", "Good"),
        "HadDiabetes": prof.get("HadDiabetes", "No"),
        "AgeCategory": prof.get("AgeCategory", "50-54"),
        "Sex": prof.get("Sex", "Female"),
        "AlcoholDrinkers": prof.get("AlcoholDrinkers", "No"),
        "SmokerStatus": prof.get("SmokerStatus", "Never smoked"),
    }

# Tek tek özellikleri değiştirerek kişiye özel “hangi faktör riski nasıl etkiliyor?” bar grafiği için farkları hesapla
def compute_feature_deltas(model, feature_order, base_feats: dict):
    import pandas as _pd
    deltas = []  # (label, delta_pp)
    base_prob = _predict_prob_from_dict(model, feature_order, base_feats)

    # Kategorikler için “daha güvenli” hedefe çekip delta ölç
    categorical_targets = {
        "SmokerStatus": "Never smoked",
        "AlcoholDrinkers": "No",
        "PhysicalActivities": "Yes",
        "GeneralHealth": "Excellent",  # daha iyi sağlık
        "HadDiabetes": "No",
    }
    for k, target in categorical_targets.items():
        if k in base_feats:
            trial = base_feats.copy()
            trial[k] = target
            p = _predict_prob_from_dict(model, feature_order, trial)
            deltas.append((k, (p - base_prob)*100.0))  # yüzde puan farkı

    # Sayısallar için küçük müdahale senaryosu
    numeric_shifts = {
        "BMI": -2.0,          # 2 BMI düşüşü
        "SleepHours": +1.0,   # 1 saat fazla uyku
    }
    for k, shift in numeric_shifts.items():
        if k in base_feats:
            trial = base_feats.copy()
            trial[k] = float(trial[k]) + shift
            p = _predict_prob_from_dict(model, feature_order, trial)
            deltas.append((f"{k} ({'+' if shift>0 else ''}{shift})", (p - base_prob)*100.0))

    # Etiketleri kullanıcı-dostu çevir
    label_map = {
        "SmokerStatus": "Switch to Never smoked",
        "AlcoholDrinkers": "Reduce heavy alcohol (No)",
        "PhysicalActivities": "Be active (Yes)",
        "GeneralHealth": "Improve general health (→ Excellent)",
        "HadDiabetes": "No diabetes",
        "BMI (-2.0)": "Lose ~2 BMI",
        "SleepHours (+1.0)": "Sleep +1 hour",
    }
    # DataFrame
    _df = _pd.DataFrame(deltas, columns=["Action", "DeltaPP"])
    _df["Action"] = _df["Action"].map(lambda x: label_map.get(x, x))
    # En çok azaltandan artana sırala
    _df = _df.sort_values("DeltaPP").reset_index(drop=True)
    return base_prob, _df

# Günlük ipucu (risk katmanına göre), her gün değişsin ama sabit kalsın
def daily_tip(tier: str):
    tips = {
        "High": [
            "🧂 Keep sodium < 1.5 g/day; check labels and avoid processed meats.",
            "🚶 10–20 min low-impact walks 5–6×/week; break sitting every 30–45 min.",
            "🥣 Aim 25–35 g/day soluble fiber (oats, barley, legumes).",
        ],
        "Medium": [
            "🐟 Eat fatty fish 2×/week and legumes 3×/week.",
            "🧘 5–10 min daily mobility + posture work supports BP and stress.",
            "🌾 Swap refined grains for whole grains (oats, brown rice, quinoa).",
        ],
        "Low": [
            "🥗 Keep Mediterranean/DASH variety; add 1 new veggie recipe weekly.",
            "👟 Hit ≥150 min/week moderate cardio + 2× strength training.",
            "💧 Prioritize hydration; limit late-day caffeine for sleep quality.",
        ],
    }
    pool = tips.get(tier, tips["Low"])
    # deterministic choice per day
    idx = _dt.date.today().toordinal() % len(pool)
    return pool[idx]

# --- BMI gauge helpers ---
BMI_BANDS = [
    ("Underweight", 10.0, 18.5, "#6aa9ff"),
    ("Normal",      18.5, 24.9, "#38c172"),
    ("Overweight",  25.0, 29.9, "#ffd166"),
    ("Obesity I",   30.0, 34.9, "#f29e4c"),
    ("Obesity II+", 35.0, 50.0, "#ef476f"),
]

def bmi_category(bmi: float):
    for name, lo, hi, color in BMI_BANDS:
        if lo <= bmi <= hi:
            return name, color
    return "Out of range", "#999"

def bmi_gauge_half(bmi: float):
    bmi = max(10, min(50, float(bmi)))
    steps = [{"range":[lo, hi], "color":color} for _, lo, hi, color in BMI_BANDS]
    cat, color = bmi_category(bmi)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        number={"suffix":" BMI"},
        gauge={
            "shape": "angular",                 # <- yarım daire
            "axis": {"range":[10,50]},
            "bar": {"color": color},
            "steps": steps,
            "threshold": {"line":{"color":color,"width":4},"thickness":0.85,"value":bmi}
        }
    ))
    fig.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10))
    return fig, cat, color

EXPLAINS = {
    "Underweight": "Focus on nutrient-dense meals and adequate protein. A small calorie surplus plus supervised strength work can help.",
    "Normal": "Great! Maintain a Mediterranean/DASH style diet and consistent activity to protect long-term cardiovascular health.",
    "Overweight": "Aim for a gentle 300–500 kcal/day deficit, increase fiber (oats/legumes), and reduce refined sugars/ultra-processed foods.",
    "Obesity I": "Prioritize low-sodium, high-fiber meals; replace saturated fats with olive oil; progress exercise gradually.",
    "Obesity II+": "Structured meal planning and clinical/dietitian support are advised for safe, sustainable changes.",
    "Out of range": "Please double-check height/weight; if correct, seek personalized medical guidance."
}

# ============== PAGE CONFIG ==============
st.set_page_config(page_title="MediSense", layout="wide")

# ============== LOTTIE BANNER ==============
def load_lottie(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

lottie = load_lottie("assets/heart_banner.json")
if lottie:
    from streamlit_lottie import st_lottie
    st_lottie(lottie, height=160, key="banner")

st.markdown(
    "<h1 style='text-align:center; color:#FFF5F5;'>MediSense — Heart Health Risk & Wellness Guide</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# ============== LOAD MODEL & DATA ==============
model = joblib.load("artifacts/model.joblib")
feature_order = joblib.load("artifacts/feature_order.joblib")
df = pd.read_csv("data/heart.csv", low_memory=False)

# For map: HadHeartAttack -> numeric (0/1)
if "HadHeartAttack" in df.columns and df["HadHeartAttack"].dtype == object:
    df["HadHeartAttack"] = df["HadHeartAttack"].map({"Yes": 1, "No": 0})

# State name -> abbreviation
STATE_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT',
    'Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL',
    'Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA',
    'Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV',
    'New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
    'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV',
    'Wisconsin':'WI','Wyoming':'WY','Guam':'GU','Puerto Rico':'PR'
}

# ============== SIDEBAR NAV ==============
with st.sidebar:
    st.markdown("### 💡 Daily Heart Tip")
    if "risk_tier" in st.session_state:
        st.info(daily_tip(st.session_state["risk_tier"]))
    else:
        st.info("Calculate your risk on the Home page to get a personalized tip.")

menu = st.sidebar.radio(
    "📌 Navigate",
    [
        "❤️ Home – Predict Risk",
        "🏃‍♂️ Exercise & Video",
        "🥗 Food Recommendations",
        "🗺️ US Risk Map",
        "🔎 Risk Factors",
        "📄 Personal Report"
    ]
)


# ============== HOME – PREDICT ==============
if menu == "❤️ Home – Predict Risk":
    st.subheader("Enter Your Health Information")

    with st.form("input_form"):
        age_cat = st.selectbox(
            "Age Category",
            ["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59",
             "60-64","65-69","70-74","75-79","80 or older"]
        )
        sex = st.selectbox("Sex", ["Female", "Male"])

        c1, c2 = st.columns(2)
        with c1:
            height_cm = st.number_input("Height (cm)", min_value=120, max_value=220, value=170, step=1)
        with c2:
            weight_kg = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=70.0, step=0.1)
        BMI = round(weight_kg / ((height_cm/100)**2), 1)
        st.metric("Calculated BMI", BMI)

        physical = st.selectbox("Physical Activity in last 30 days?", ["No","Yes"])
        sleep_hours = st.number_input("Sleep Hours (per day)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        gen_health = st.selectbox("General Health", ["Excellent","Very good","Good","Fair","Poor"])
        had_diabetes = st.selectbox("Diabetes", ["No","Yes","No, borderline diabetes","Yes (during pregnancy)"])
        alcohol = st.selectbox("Heavy Alcohol Consumption", ["No","Yes"])
        smoker = st.selectbox("Smoking Status",
                              ["Never smoked","Former smoker","Current smoker (some days)","Current smoker (every day)"])

        # Mental health (only for tips, NOT in model)
        stress_level = st.selectbox("Stress Level", ["Low","Moderate","High"])

        state = st.selectbox("State (used in Map page)", sorted(df["State"].dropna().unique()))

        submit = st.form_submit_button("Predict Risk")

    if submit:
        row = pd.DataFrame([{
            "BMI": BMI,
            "PhysicalActivities": physical,
            "SleepHours": sleep_hours,
            "GeneralHealth": gen_health,
            "HadDiabetes": had_diabetes,
            "AgeCategory": age_cat,
            "Sex": sex,
            "AlcoholDrinkers": alcohol,
            "SmokerStatus": smoker
        }], columns=feature_order)

        st.session_state["state_selected"] = state

        prob = float(model.predict_proba(row)[:, 1])
        risk_pct = round(prob * 100, 1)

        if   risk_pct >= 60: tier = "High"
        elif risk_pct >= 30: tier = "Medium"
        else:                tier = "Low"

        # --- Prediction Result (stable layout) ---
        st.subheader("Prediction Result")

        # 1) Üst satır: 3 sabit metrik (kayma olmasın)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Estimated Risk", f"{risk_pct:.1f} %")
        with mc2:
            badge = "🟥 High" if tier == "High" else ("🟧 Medium" if tier == "Medium" else "🟩 Low")
            st.metric("Tier", tier)  # metrik kısa, rozet aşağıdaki satıra
        with mc3:
            st.metric("State", state)

        st.caption("Educational output — not medical advice.")
        st.write("")  # küçük boşluk

        # 2) Session state (diğer sayfalar için)
        st.session_state["risk_prob"] = prob
        st.session_state["risk_pct"] = risk_pct
        st.session_state["risk_tier"] = tier
        st.session_state["risk_level"] = badge
        st.session_state["state_selected"] = state
        st.session_state["profile"] = {
            "BMI": BMI,
            "GeneralHealth": gen_health,
            "AlcoholDrinkers": alcohol,
            "HadDiabetes": had_diabetes,
            "SmokerStatus": smoker,
            "StressLevel": stress_level,
            "PhysicalActivities": physical,
            "SleepHours": sleep_hours,
            "AgeCategory": age_cat,
            "Sex": sex
        }

        # 3) Alt satır: BMI gauge + poster (ayrı blok — kayma yapmasın)
        st.markdown("### BMI & Guidance")
        fig, cat, clr = bmi_gauge_half(BMI)
        g1, g2 = st.columns([1.2, 1])
        with g1:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"**BMI:** {BMI:.1f} — **Category:** "
                f"<span style='color:{clr};font-weight:700'>{cat}</span>",
                unsafe_allow_html=True
            )
            st.caption(EXPLAINS.get(cat, EXPLAINS["Out of range"]))
        with g2:
            if os.path.exists("assets/bmi_chart.jpg"):
                st.image("assets/bmi_chart.jpg", caption="BMI Classification (WHO style)", use_column_width=True)
            else:
                st.info("Add BMI poster to: assets/bmi_chart.jpg")



# ============== EXERCISE & VIDEO ==============
elif menu == "🏃‍♂️ Exercise & Video":
    st.subheader("🏋️ Heart-Healthy Exercise Recommendations & Videos")

    if "risk_tier" not in st.session_state:
        st.warning("Please go to **❤️ Home – Predict Risk** first.")
    else:
        tier = st.session_state["risk_tier"]
        risk_pct = st.session_state["risk_pct"]
        badge = "🟥 High Risk" if tier=="High" else ("🟧 Medium Risk" if tier=="Medium" else "🟩 Low Risk")
        st.write(f"**Estimated Risk:** {risk_pct:.1f}% — {badge}")

        # Video suggestion per tier
        video_bank = {
            "Low":    ("30-min Low Impact Cardio", "https://www.youtube.com/watch?v=ml6cT4AZdqI"),
            "Medium": ("Moderate Cardio + Strength", "https://www.youtube.com/watch?v=UItWltVZZmE"),
            "High":   ("Gentle Chair Exercises", "https://www.youtube.com/watch?v=8BcPHWGQO44"),
        }
        title, url = video_bank["High" if tier=="High" else ("Medium" if tier=="Medium" else "Low")]
        st.markdown(f"**Suggested Exercise Video:** {title}")
        try:
            st.video(url)
        except Exception:
            st.link_button("Open on YouTube ▶️", url)

        # Long guidance (do & avoid)
        with st.expander("✅ Weekly Plan, Why It Matters & What to Avoid"):
            if tier == "High":
                st.markdown("""
**🎯 First 4–6 weeks goal:** build habit safely, improve circulation, reduce BP and symptoms.

**Do (weekly plan)**
- 🚶‍♂️ **Cardio (5–6×/week):** 10–20 min **low-impact** (easy walk, stationary bike, water walking).
- 🧘 **Mobility (daily 5–10 min):** chest, hips, calves + breathing drills.
- 🏋️ **Strength (2×/week):** very light bands/body-weight (sit-to-stand, wall push-ups), 1–2 sets × 10–12 reps.
- ⏱️ Break up sitting every **30–45 min** (stand/walk 1–2 min).

**Avoid / Use caution**
- ❌ High-intensity intervals, sprints, heavy lifting with Valsalva (breath holding).
- ❌ “No-pain-no-gain” mindset; stop with chest pain, dizziness, unusual breathlessness.
- ❌ Dehydration; keep water handy.

**Why it matters**
- 🫀 Regular activity improves endothelial function, lowers resting HR/BP, and reduces triglycerides.
- 🧠 Low-impact training also decreases stress and anxiety, indirectly protecting the heart.
                """)
            elif tier == "Medium":
                st.markdown("""
**🎯 12-week goal:** reach guideline levels and maintain.

**Do (weekly plan)**
- 🚶 **Cardio (4–5×/week):** **≥150 min/week** moderate intensity (brisk walk/cycle), e.g., 5×30 min.
- 🏋️ **Strength (2–3×/week):** full-body (squat/lunge/row/press), **2–3 sets × 8–12 reps**.
- 🧎 **Core & Balance (2×/week):** plank variations, bird-dog, side steps.
- 🧘 **Mobility (daily 5–10 min)**; posture work for neck/upper back.

**Avoid / Use caution**
- ⚠️ Sudden jumps to vigorous intensity; progress by **+10–15%** every 2–3 weeks.
- ⚠️ Long uninterrupted sitting; micro-breaks each hour.

**Why it matters**
- 📉 Aerobic + resistance training improves lipid profile, insulin sensitivity and reduces BP.
                """)
            else:  # Low
                st.markdown("""
**🎯 Goal:** maintain fitness; add variety.

**Do (weekly plan)**
- 🏃 **Cardio (3–4×/week):** 25–35 min moderate; optional intervals (3 min easy / 1 min brisk × 6–8).
- 🏋️ **Strength (2×/week):** full-body **2–3 sets × 8–12**; control tempo.
- 🧘 **Mobility (daily 5–10 min)**.

**Avoid / Use caution**
- ⚠️ Big spikes in training load (keep RPE around 5–7/10).
- ⚠️ Excess late-day caffeine that impairs sleep (sleep is heart-protective).

**Why it matters**
- 🫀 Consistency maintains vascular health, supports weight control and stress resilience.
                """)




# ============== FOOD RECOMMENDATIONS ==============
elif menu == "🥗 Food Recommendations":
    import os
    st.subheader("🥗 Heart-Healthy Food Recommendations")

    if "risk_tier" not in st.session_state:
        st.warning("Please go to **❤️ Home – Predict Risk** first.")
    else:
        tier = st.session_state["risk_tier"]
        badge = "🟥 High" if tier=="High" else ("🟧 Medium" if tier=="Medium" else "🟩 Low")
        st.markdown(f"**Your current tier:** {badge}")

        # ---- Rich intro (why these foods) ----
        st.markdown("""
A heart-healthy pattern emphasizes **omega-3s**, **whole grains**, **vegetables**, **healthy fats**, and **low sodium**.
This page is **personalized by your risk tier** — we highlight foods most likely to help **you** first.

**How to build your plate :**
- 🥗 **½ plate vegetables** (leafy greens, cruciferous veg).
- 🌾 **¼ plate whole grains** (oats, quinoa, brown rice).
- 🍗 **¼ plate lean protein** (fish/legumes/poultry).
- 🫒 **Add healthy fats** (olive oil, avocado, **unsalted** nuts).
- 🧂 **Lower sodium**: avoid processed meats, canned soups, salty snacks; **read labels**.
""")

        # (Optional) quick wins by tier
        with st.expander("💡 Quick wins for your tier"):
            if tier == "High":
                st.write(
                    "- Swap processed meats for **beans/fish**.\n"
                    "- Cook with **olive oil** instead of butter.\n"
                    "- Choose **low-sodium** items; rinse canned foods.\n"
                    "- Aim **25–35 g soluble fiber/day** (oats, barley, legumes)."
                )
            elif tier == "Medium":
                st.write(
                    "- Add **1 extra veggie** serving daily.\n"
                    "- **Fish 2×/week** or **walnuts/chia** on non-fish days.\n"
                    "- Replace white bread with **whole-grain**."
                )
            else:
                st.write(
                    "- Keep variety: rotate leafy greens & cruciferous veg.\n"
                    "- **Unsalted nuts** for snacks.\n"
                    "- Keep sugary drinks for rare occasions."
                )

        # ---- Local gallery (risk-filtered tabs) ----
        FOODS = {
            "🐟 Omega-3 Sources": [
                ("assets/foods/salmon.jpg",        "Salmon — omega-3 lowers triglycerides, supports rhythm"),
                ("assets/foods/mackerel.jpg",      "Mackerel/Sardines — raises HDL, anti-inflammatory"),
                ("assets/foods/walnuts.jpg",       "Walnuts — plant omega-3 (ALA) + polyphenols"),
                ("assets/foods/chia_seeds.jpg",    "Chia/Flax — fiber + omega-3 for lipid control"),
            ],
            "🌾 Whole Grains": [
                ("assets/foods/oats.jpg",          "Oats/Barley — beta-glucan reduces LDL cholesterol"),
                ("assets/foods/quinoa.jpg",        "Quinoa — complete protein, magnesium for rhythm"),
                ("assets/foods/brown_rice.jpg",    "Brown Rice — fiber for BP & glycemic control"),
            ],
            "🥦 Vegetables": [
                ("assets/foods/leafy_greens.jpg",  "Leafy Greens — nitrates & folate support vessels"),
                ("assets/foods/broccoli.jpg",      "Broccoli — antioxidant / anti-inflammatory"),
                ("assets/foods/mixed_veggies.jpg", "Mixed Vegetables — color diversity = nutrient diversity"),
            ],
            "🥑 Healthy Fats": [
                ("assets/foods/olive_oil.jpg",     "Olive Oil — MUFAs + polyphenols for endothelial health"),
                ("assets/foods/avocado.jpg",       "Avocado — potassium + MUFAs for BP/cholesterol"),
                ("assets/foods/nuts.jpg",          "Mixed Nuts (unsalted) — satiety + HDL benefits"),
            ],
            "🍓 Fruits & Antioxidants": [
                ("assets/foods/berries.jpg",       "Berries — anthocyanins lower BP & oxidative stress"),
                ("assets/foods/tomatoes.jpg",      "Tomatoes — lycopene linked to lower LDL"),
                ("assets/foods/citrus_fruits.jpg", "Citrus — vitamin C & flavonoids for vessel health"),
            ],
        }

        # Always apply risk filter (Show all removed)
        FILTERS = {
            "High": {
                "🐟 Omega-3 Sources": ["salmon.jpg","mackerel.jpg","chia_seeds.jpg","walnuts.jpg"],
                "🌾 Whole Grains":    ["oats.jpg","quinoa.jpg","brown_rice.jpg"],
                "🥦 Vegetables":      ["leafy_greens.jpg","broccoli.jpg"],
                "🥑 Healthy Fats":    ["olive_oil.jpg","avocado.jpg","nuts.jpg"],
                "🍓 Fruits & Antioxidants": ["berries.jpg","tomatoes.jpg","citrus_fruits.jpg"],
            },
            "Medium": {
                "🐟 Omega-3 Sources": ["salmon.jpg","mackerel.jpg","walnuts.jpg","chia_seeds.jpg"],
                "🌾 Whole Grains":    ["oats.jpg","quinoa.jpg","brown_rice.jpg"],
                "🥦 Vegetables":      ["leafy_greens.jpg","broccoli.jpg","mixed_veggies.jpg"],
                "🥑 Healthy Fats":    ["olive_oil.jpg","avocado.jpg","nuts.jpg"],
                "🍓 Fruits & Antioxidants": ["berries.jpg","tomatoes.jpg","citrus_fruits.jpg"],
            },
            "Low": "ALL"
        }

        def render_grid(items):
            cols = st.columns(2)
            for i, (path, caption) in enumerate(items):
                with cols[i % 2]:
                    if os.path.exists(path):
                        st.image(path, caption=caption, use_column_width=True)
                    else:
                        st.info(f"Missing: {os.path.basename(path)}")

        tabs = st.tabs(list(FOODS.keys()))
        for tab, group in zip(tabs, FOODS.keys()):
            with tab:
                items = FOODS[group]
                if FILTERS.get(tier) != "ALL":
                    allowed = set(FILTERS[tier].get(group, []))
                    items = [x for x in items if os.path.basename(x[0]) in allowed]
                render_grid(items)

        # ---- Long, risk-based tips (Do / Avoid / Why) ----
        st.markdown("### 📌 Risk-Based Dietary Guidance")
        if tier == "High":
            st.markdown("""
**Do**
- 🐟 **Omega-3 twice/week** (salmon/sardines) or daily plant sources (walnuts, chia/flax).
- 🥣 **25–35 g/day soluble fiber** (oats, barley, legumes) to lower LDL.
- 🥗 Plate model: **½ vegetables**, ¼ whole grain, ¼ lean protein.
- 🫒 Replace butter/cream with **olive oil/avocado**; choose **unsalted nuts**.

**Avoid / Limit**
- 🧂 **Sodium < 1.5 g/day** (≈ 3.8 g salt). Check labels; avoid processed meats/soups.
- 🧁 Sugary drinks & refined carbs (white bread/pastries).
- 🧈 **Trans fats** and high saturated fat (deep-fried/processed).

**Why it matters**
- Lower sodium reduces BP; soluble fiber reduces LDL; healthy fats improve endothelial function.
            """)
        elif tier == "Medium":
            st.markdown("""
**Do**
- 🌾 **2–3 servings/day** whole grains; swap white rice/bread for brown/quinoa.
- 🐟 **Fish 2×/week**; legumes **3×/week**; vegetables **5+ servings/day**.
- 🫒 Use olive oil as primary added fat; prefer **unsalted** nuts.

**Avoid / Limit**
- 🧂 Highly salted processed foods; choose **low-sodium (<140 mg/serving)**.
- 🍩 Ultra-processed snacks; limit desserts to special occasions.

**Why it matters**
- Moderate changes in fiber/fat quality and sodium produce measurable BP and lipid improvements.
            """)
        else:
            st.markdown("""
**Do**
- Keep **Mediterranean/DASH** pattern: diverse vegetables/fruits, whole grains, lean proteins.
- Rotate **new veggie-based recipes weekly** to maintain variety.
- Maintain hydration and consistent meal timing.

**Avoid / Limit**
- Gradual drift toward processed, salty convenience foods — keep your current standards.

**Why it matters**
- Consistency preserves favorable lipid/BP profile and reduces long-term atherosclerotic risk.
            """)

        with st.expander("📄 Food-by-Food: How Each Supports Heart Health"):
            st.markdown("""
- **Salmon / Mackerel / Sardines**: marine omega-3s (EPA/DHA) reduce triglycerides and arrhythmia risk.
- **Walnuts / Chia / Flax**: plant omega-3 (ALA) + fiber → LDL reduction and anti-inflammatory effects.
- **Oats / Barley**: beta-glucan binds bile acids, lowering LDL cholesterol.
- **Leafy Greens**: dietary nitrates improve endothelial function and nitric oxide availability.
- **Tomatoes**: lycopene correlates with reduced LDL oxidation.
- **Olive Oil**: monounsaturated fat and polyphenols improve HDL function and vascular health.
- **Berries / Citrus**: polyphenols and vitamin C combat oxidative stress and support vessel integrity.
            """)





# ============== US RISK MAP (Your risk vs state & nation, rank & percentile) ==============
elif menu == "🗺️ US Risk Map":
    st.subheader("🗺️ Heart Attack Prevalence by State (BRFSS %) — vs Your Estimated Risk")

    # Guardrails
    if "State" not in df.columns or "HadHeartAttack" not in df.columns:
        st.warning("`State` / `HadHeartAttack` columns not found in dataset.")
    elif "risk_pct" not in st.session_state:
        st.warning("Please go to **❤️ Home – Predict Risk** first and get your personal estimate.")
    else:
        # ---- Prepare data
        sdf = df[["State", "HadHeartAttack"]].dropna()
        sdf["HadHeartAttack"] = pd.to_numeric(sdf["HadHeartAttack"], errors="coerce")
        sdf = sdf.dropna(subset=["HadHeartAttack"])

        state_rates = sdf.groupby("State", as_index=False)["HadHeartAttack"].mean()
        state_rates["RatePct"] = state_rates["HadHeartAttack"] * 100.0
        state_rates["abbr"] = state_rates["State"].map(STATE_ABBR)
        state_rates = state_rates.dropna(subset=["abbr"])

        if state_rates.empty:
            st.warning("No state-level data available after cleaning.")
        else:
            # National average
            nat_pct = float(state_rates["RatePct"].mean())

            # User state & personal risk
            user_state = st.session_state.get("state_selected", None)
            if not user_state or user_state not in state_rates["State"].values:
                user_state = state_rates.iloc[0]["State"]  # fallback
            user_pct = float(st.session_state["risk_pct"])

            srow = state_rates[state_rates["State"] == user_state].iloc[0]
            state_pct = float(srow["RatePct"])

            # Rank & percentile (lower prevalence = better rank)
            state_rates_sorted = state_rates.sort_values("RatePct").reset_index(drop=True)
            total_regions = len(state_rates_sorted)
            rank_pos = int(state_rates_sorted.index[state_rates_sorted["State"] == user_state][0]) + 1
            percentile = round(100.0 * rank_pos / total_regions, 1)  # e.g., 65.0 → higher than ~65% of states

            # ---- Header metrics (stable 4 columns)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Your Risk (model)", f"{user_pct:.1f} %")
            m2.metric(f"{user_state} Avg (BRFSS)", f"{state_pct:.1f} %", delta=f"{user_pct - state_pct:+.1f} pp")
            m3.metric("USA Avg (BRFSS)", f"{nat_pct:.1f} %", delta=f"{user_pct - nat_pct:+.1f} pp")
            m4.metric("State Rank / Percentile", f"{rank_pos}/{total_regions}",
                      help=f"{percentile}th percentile = prevalence higher than ~{percentile}% of states")
            st.caption("‘pp’ = percentage points. BRFSS = population prevalence; your value is a personal model estimate.")

            # ---- Choropleth with user's state highlighted
            import plotly.graph_objects as go

            base = go.Choropleth(
                locations=state_rates["abbr"],
                z=state_rates["RatePct"],
                locationmode="USA-states",
                colorscale="Reds",
                colorbar_title="Prevalence (%)",
                marker_line_color="white",
                marker_line_width=0.8,
                hovertext=state_rates["State"],
                hovertemplate="<b>%{hovertext}</b><br>Prevalence: %{z:.1f}%<extra></extra>",
            )

            highlight = go.Choropleth(
                locations=[srow["abbr"]],
                z=[srow["RatePct"]],
                locationmode="USA-states",
                colorscale="Reds",
                showscale=False,
                marker_line_color="black",
                marker_line_width=3.0,   # thicker border to highlight
                hovertext=[user_state],
                hovertemplate="<b>%{hovertext}</b><br>Prevalence: %{z:.1f}% (highlighted)<extra></extra>",
            )

            fig = go.Figure(data=[base, highlight])
            fig.update_layout(
                geo=dict(scope="usa"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=520,
                title="State Prevalence vs Your Estimated Risk"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---- Short interpretation under the map
            delta_state = user_pct - state_pct
            if delta_state >= 2:
                st.warning(f"Your personal risk is **{delta_state:.1f} pp higher** than the **{user_state}** average.")
            elif delta_state <= -2:
                st.success(f"Your personal risk is **{abs(delta_state):.1f} pp lower** than the **{user_state}** average — great!")
            else:
                st.info(f"Your personal risk is **similar** to the **{user_state}** average (±2 pp).")

            st.caption("The map shows BRFSS-based state prevalence; your risk is a personalized model estimate and may differ from population averages.")



# ===================== RISK FACTORS (text-only, super clear) =====================
elif menu == "🔎 Risk Factors":
    st.subheader("🔎 What should you change first? (No chart, just clear priorities)")

    if "risk_tier" not in st.session_state or "profile" not in st.session_state:
        st.warning("Please go to **❤️ Home – Predict Risk** and calculate your risk first.")
    else:
        base_feats = _current_features_for_model()
        base_prob, df_imp = compute_feature_deltas(model, feature_order, base_feats)  # Action, DeltaPP

        # Negatif = risk AZALIR (iyi). Pozitif = risk ARTAR (kötü).
        df_imp = df_imp.sort_values("DeltaPP")
        top_improvements = df_imp.head(5)  # en çok azaltan 5
        worst_riskers = df_imp.sort_values("DeltaPP", ascending=False).head(3)  # en çok artıran 3

        st.markdown(f"**Estimated Risk (current): {base_prob*100:.1f}%**")

        st.markdown("### ✅ Highest-impact improvements")
        lines = [f"- {row['Action']} → **{abs(row['DeltaPP']):.1f} pp lower** (estimated)"
                 for _, row in top_improvements.iterrows()]
        st.write("\n".join(lines) if lines else "- No clear improvement items found.")

        st.markdown("### ⚠️ Things that would raise your risk (avoid)")
        lines2 = [f"- {row['Action']} → **+{row['DeltaPP']:.1f} pp** higher (estimated)"
                  for _, row in worst_riskers.iterrows() if row['DeltaPP'] > 0]
        st.write("\n".join(lines2) if lines2 else "- No major worsening factors detected.")

        st.caption("‘pp’ = percentage points (e.g., 12.0% → 9.5% is −2.5 pp). Single-factor estimates with others held constant. Educational — not medical advice.")



elif menu == "📄 Personal Report":
    st.subheader("📄 Your Personalized Heart Health Report")

    if "risk_tier" not in st.session_state or "profile" not in st.session_state:
        st.warning("Please go to **❤️ Home – Predict Risk** first.")
    else:
        tier = st.session_state["risk_tier"]
        badge = tier_badge(tier)
        pct = st.session_state["risk_pct"]
        prof = st.session_state["profile"].copy()
        BMI_val = prof.get("BMI", 25.0)

        # Başlık kutuları
        c1,c2,c3 = st.columns(3)
        c1.metric("Risk", f"{pct:.1f} %", badge)
        cat, clr = bmi_category(BMI_val)
        c2.metric("BMI", f"{BMI_val:.1f}", cat)
        c3.metric("Tier", tier)

        # Özet metinler
        st.markdown("### Summary")
        st.write(f"- **Risk Tier:** {badge}")
        st.write(f"- **BMI:** {BMI_val:.1f} — **Category:** {cat}")
        st.write(f"- **General Health:** {prof.get('GeneralHealth','-')} | **Diabetes:** {prof.get('HadDiabetes','-')}")
        st.write(f"- **Physical Activity (30 days):** {prof.get('PhysicalActivities','Yes')} | **Sleep:** {prof.get('SleepHours','7.0')} h")
        st.write(f"- **Smoking:** {prof.get('SmokerStatus','Never smoked')} | **Alcohol (heavy):** {prof.get('AlcoholDrinkers','No')}")

        # Kısa plan
        st.markdown("### Action Plan (diet & exercise)")
        if tier == "High":
            st.markdown(
                "- 🧂 **Sodium <1.5 g/day**, avoid processed meats/soups.\n"
                "- 🥣 **25–35 g/day soluble fiber** (oats, barley, legumes).\n"
                "- 🐟 **Omega-3**: fatty fish 2×/week or walnuts/chia daily.\n"
                "- 🚶 10–20 min low-impact walks 5–6×/week + 2× light strength.\n"
            )
        elif tier == "Medium":
            st.markdown(
                "- 🌾 Whole grains **2–3 servings/day**; veggies **5+ servings/day**.\n"
                "- 🐟 Fish 2×/week; legumes 3×/week; olive oil as main added fat.\n"
                "- 🚶 ≥150 min/week moderate cardio + 2–3× strength.\n"
            )
        else:
            st.markdown(
                "- 🥗 Maintain Mediterranean/DASH variety; rotate new veggie recipes weekly.\n"
                "- 👟 ≥150 min/week cardio + 2× strength; good sleep & hydration.\n"
            )

        # İndirilebilir metin (MD)
        report_md = f"""# MediSense — Personal Heart Health Report
Date: {_dt.date.today().isoformat()}

**Risk:** {pct:.1f}% ({tier})
**BMI:** {BMI_val:.1f} ({cat})

## Profile
- General Health: {prof.get('GeneralHealth','-')}
- Diabetes: {prof.get('HadDiabetes','-')}
- Smoking: {prof.get('SmokerStatus','-')}
- Alcohol (heavy): {prof.get('AlcoholDrinkers','-')}
- Physical Activity (30 days): {prof.get('PhysicalActivities','-')}
- Sleep Hours: {prof.get('SleepHours','-')}

## Diet & Exercise Plan
{"- Sodium <1.5 g/day; soluble fiber 25–35 g/day; Omega-3 2×/week; low-impact walks 5–6×/week." if tier=="High" else (
 "- Whole grains 2–3/day; veggies 5+; fish 2×/week; ≥150 min/wk cardio + 2–3× strength." if tier=="Medium" else
 "- Maintain Mediterranean/DASH pattern; ≥150 min/wk cardio + 2× strength; sleep & hydration."
)}
(Educational content — not medical advice.)
"""
        st.download_button("⬇️ Download Report (Markdown)", data=report_md, file_name="medisense_report.md")







