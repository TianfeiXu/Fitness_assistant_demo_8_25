"""
Fitness AI Demo (Streamlit) - MVP
---------------------------------
One-file Streamlit app that:
  • collects user profile (height/weight/age/sex)
  • parses free-form workout logs (CN/EN) → structured rows (exercise, weight_kg, reps, sets, minutes, rpe)
  • estimates calories via METs (with fallbacks for strength training)
  • tracks 7-day trends (calories, optional volume)
  • maintains per-body-part "heat" score with exponential decay (half-life=7 days)
  • generates same-day training suggestions based on last 7 days and profile

How to run:
  1) pip install streamlit pandas altair python-dateutil openai
  2) set OPENAI_API_KEY in your environment (optional; app works with regex-only fallback):
     - Windows (PowerShell): $env:OPENAI_API_KEY="sk-..."
  3) streamlit run app.py

Notes:
  • This file is intentionally self-contained (no extra modules required).
  • If OpenAI key is missing, the regex parser runs and the app still works.
  • Customize METS, BODY_PART_MAP below as needed.
"""

import os
import re
import json
import math
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import pandas as pd
import altair as alt
from dateutil.relativedelta import relativedelta

import streamlit as st

# ---------- Optional OpenAI client (safe import) ----------
OPENAI_AVAILABLE = False
try:
    import openai  # Legacy SDK interface retained for widest compatibility
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------- Constants & Config ----------
DATA_DIR = "data"
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
WORKOUTS_CSV = os.path.join(DATA_DIR, "workouts.csv")
HEAT_CSV = os.path.join(DATA_DIR, "heat.csv")

BODY_PARTS = ["chest", "back", "legs", "shoulders", "arms", "core", "cardio", "other"]

# Exercise → primary body part mapping (extend as needed)
BODY_PART_MAP = {
    # Strength (upper)
    "bench_press": "chest",
    "incline_bench_press": "chest",
    "push_up": "chest",
    "overhead_press": "shoulders",
    "shoulder_press": "shoulders",
    "lateral_raise": "shoulders",
    "front_raise": "shoulders",
    "dumbbell_row": "back",
    "barbell_row": "back",
    "pull_up": "back",
    "chin_up": "back",
    "lat_pulldown": "back",
    "biceps_curl": "arms",
    "triceps_extension": "arms",
    # Strength (lower/core)
    "squat": "legs",
    "front_squat": "legs",
    "deadlift": "legs",
    "romanian_deadlift": "legs",
    "lunge": "legs",
    "leg_press": "legs",
    "plank": "core",
    "crunch": "core",
    "leg_raise": "core",
    # Cardio
    "run": "cardio",
    "running": "cardio",
    "treadmill": "cardio",
    "cycle": "cardio",
    "cycling": "cardio",
    "bike": "cardio",
    "rower": "cardio",
    "rowing": "cardio",
    "jump_rope": "cardio",
}

# MET values (approx) for common modalities
METS = {
    # Cardio (approx ranges; choose sensible defaults)
    "run_easy": 8.3,       # ~6 mph (9:40/mi)
    "run_hard": 11.0,      # faster running
    "cycling_easy": 6.8,   # 12–13.9 mph
    "cycling_hard": 8.0,   # 14–15.9 mph
    "rowing": 7.0,
    "jump_rope": 10.0,
    # Strength training (broad buckets)
    "strength_medium": 3.5,
    "strength_hard": 6.0,
}

HALF_LIFE_DAYS = 7  # for heat decay
HEAT_DECAY = math.exp(-math.log(2) / HALF_LIFE_DAYS)

# Thresholds for body-part heat
HEAT_LOW = 1.0
HEAT_HIGH = 3.0

# ---------- Utilities ----------
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(WORKOUTS_CSV):
        pd.DataFrame(columns=[
            "date","exercise","body_part","weight_kg","reps","sets","minutes","rpe","est_met","est_kcal","volume"
        ]).to_csv(WORKOUTS_CSV, index=False)
    if not os.path.exists(USERS_CSV):
        pd.DataFrame([{ "height_cm": 175, "weight_kg": 75, "age": 27, "sex": "male", "goal": "fat_loss" }]).to_csv(USERS_CSV, index=False)
    if not os.path.exists(HEAT_CSV):
        row = {"date": (dt.date.today() - relativedelta(days=1)).isoformat()}
        for bp in BODY_PARTS:
            row[bp] = 0.0
        pd.DataFrame([row]).to_csv(HEAT_CSV, index=False)


def load_users() -> pd.DataFrame:
    ensure_dirs()
    return pd.read_csv(USERS_CSV)


def save_user_profile(height_cm: float, weight_kg: float, age: int, sex: str, goal: str):
    df = pd.DataFrame([{ "height_cm": height_cm, "weight_kg": weight_kg, "age": age, "sex": sex, "goal": goal }])
    df.to_csv(USERS_CSV, index=False)


def load_workouts() -> pd.DataFrame:
    ensure_dirs()
    df = pd.read_csv(WORKOUTS_CSV)
    # sanitize types
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")
        df["reps"] = pd.to_numeric(df["reps"], errors="coerce")
        df["sets"] = pd.to_numeric(df["sets"], errors="coerce")
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
        df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce")
        df["est_met"] = pd.to_numeric(df["est_met"], errors="coerce")
        df["est_kcal"] = pd.to_numeric(df["est_kcal"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df


def append_workouts(rows: List[Dict[str, Any]]):
    if not rows:
        return
    df_old = load_workouts()
    df_new = pd.DataFrame(rows)
    df = pd.concat([df_old, df_new], ignore_index=True)
    df.to_csv(WORKOUTS_CSV, index=False)


def load_heat() -> pd.DataFrame:
    ensure_dirs()
    df = pd.read_csv(HEAT_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def save_heat(df: pd.DataFrame):
    df.to_csv(HEAT_CSV, index=False)


# ---------- Parsing ----------
@dataclass
class WorkoutItem:
    date: str
    exercise: str
    body_part: str
    weight_kg: float
    reps: int
    sets: int
    minutes: float
    rpe: float


def to_snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unknown"


def map_body_part(exercise: str) -> str:
    key = to_snake(exercise)
    return BODY_PART_MAP.get(key, "other")


# Regex fallback parser for common patterns
RE_WEIGHT_REPS_SETS = re.compile(r"(?P<exercise>[\u4e00-\u9fa5A-Za-z_\s]+?)\s*(?P<weight>\d{1,3})\s*kg\s*([x×*])\s*(?P<reps>\d{1,3})\s*(?:[x×*]\s*(?P<sets>\d{1,2}))?", re.IGNORECASE)
RE_REPS_SETS = re.compile(r"(?P<exercise>[\u4e00-\u9fa5A-Za-z_\s]+?)\s*(?P<reps>\d{1,3})\s*(?:次|reps?)\s*[x×*]\s*(?P<sets>\d{1,2})", re.IGNORECASE)
RE_MINUTES = re.compile(r"(?P<exercise>[\u4e00-\u9fa5A-Za-z_\s]*?)(?:跑步|running|treadmill|cycle|cycling|bike|row(?:ing|er)?|jump(?:\s*rope)?)\D*(?P<minutes>\d{1,3})\s*(?:分钟|min|minutes?)", re.IGNORECASE)
RE_RPE = re.compile(r"RPE\s*(?P<rpe>\d{1,2})", re.IGNORECASE)


def regex_parse(text: str, default_date: dt.date) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    text = text.replace("；", ";").replace("，", ",")

    # Cardio minutes
    for m in RE_MINUTES.finditer(text):
        raw_ex = m.group("exercise").strip() or "running"
        minutes = float(m.group("minutes"))
        rpe = None
        items.append({
            "date": default_date.isoformat(),
            "exercise": to_snake(raw_ex if raw_ex else "running"),
            "body_part": map_body_part(raw_ex if raw_ex else "running"),
            "weight_kg": None,
            "reps": None,
            "sets": None,
            "minutes": minutes,
            "rpe": rpe,
        })

    # Weight kg × reps × sets
    for m in RE_WEIGHT_REPS_SETS.finditer(text):
        ex = to_snake(m.group("exercise"))
        weight = float(m.group("weight"))
        reps = int(m.group("reps"))
        sets = int(m.group("sets") or 3)
        rpe = None
        m_rpe = RE_RPE.search(text)
        if m_rpe:
            try:
                rpe = float(m_rpe.group("rpe"))
            except Exception:
                rpe = None
        items.append({
            "date": default_date.isoformat(),
            "exercise": ex,
            "body_part": map_body_part(ex),
            "weight_kg": weight,
            "reps": reps,
            "sets": sets,
            "minutes": None,  # will be estimated later
            "rpe": rpe,
        })

    # reps×sets without weight
    for m in RE_REPS_SETS.finditer(text):
        ex = to_snake(m.group("exercise"))
        reps = int(m.group("reps"))
        sets = int(m.group("sets"))
        rpe = None
        items.append({
            "date": default_date.isoformat(),
            "exercise": ex,
            "body_part": map_body_part(ex),
            "weight_kg": None,
            "reps": reps,
            "sets": sets,
            "minutes": None,
            "rpe": rpe,
        })

    # deduplicate rough overlaps by exercise signature
    # (keep the first occurrence)
    seen = set()
    dedup = []
    for it in items:
        sig = (it["exercise"], it.get("weight_kg"), it.get("reps"), it.get("sets"), it.get("minutes"))
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append(it)
    return dedup


LLM_SYSTEM = (
    "You are a fitness log parser. Return ONLY JSON in UTF-8. "
    "Parse Chinese/English workout notes into a list of items with fields: "
    "date (YYYY-MM-DD), exercise (snake_case), body_part (one of chest/back/legs/shoulders/arms/core/cardio/other), "
    "weight_kg (number or null), reps (int or null), sets (int or null), minutes (number or null), rpe (number or null). "
    "Infer body_part from exercise; if unsure, 'other'. Default date is today if missing."
)

LLM_USER_TMPL = """
Text:
{log}
Today: {today}
Output schema:
{{"items": [{{"date":"YYYY-MM-DD","exercise":"bench_press","body_part":"chest","weight_kg":60,"reps":5,"sets":3,"minutes":null,"rpe":8}}]}}
"""


def llm_parse(text: str, today: dt.date) -> List[Dict[str, Any]]:
    if not OPENAI_AVAILABLE:
        return []
    try:
        prompt_user = LLM_USER_TMPL.format(log=text, today=today.isoformat())
        # Use Chat Completions for broad compatibility
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.1,
        )
        content = resp["choices"][0]["message"]["content"]
        data = json.loads(content)
        items = data.get("items", [])
        # fix-ups
        for it in items:
            it["exercise"] = to_snake(str(it.get("exercise","")))
            it["body_part"] = it.get("body_part") or map_body_part(it["exercise"])
            it["date"] = (it.get("date") or today.isoformat())
        return items
    except Exception:
        return []


# ---------- Energy & Heat Calculation ----------

def estimate_minutes(item: Dict[str, Any]) -> float:
    # For strength with no minutes, estimate 2 min / set as a simple baseline
    if item.get("minutes") not in (None, "", float("nan")):
        try:
            return float(item["minutes"]) or 0.0
        except Exception:
            pass
    sets = float(item.get("sets") or 0)
    return round(2.0 * sets, 1)


def pick_strength_met(rpe: float, exercise: str) -> float:
    if rpe is None:
        rpe = 7.0
    ex = to_snake(exercise)
    hard_compounds = {"squat", "deadlift", "bench_press", "overhead_press", "barbell_row"}
    if rpe >= 8.0 or ex in hard_compounds:
        return METS["strength_hard"]
    return METS["strength_medium"]


def estimate_met(item: Dict[str, Any]) -> float:
    bp = item.get("body_part", "other")
    ex = to_snake(item.get("exercise", ""))
    if bp == "cardio" or ex in {"run","running","treadmill","cycle","cycling","bike","rower","rowing","jump_rope"}:
        # Choose simple defaults; you can expand with speed/RPE mapping later
        if ex in {"running","run","treadmill"}:
            return METS["run_easy"]
        if ex in {"cycling","cycle","bike"}:
            return METS["cycling_easy"]
        if ex in {"rower","rowing"}:
            return METS["rowing"]
        if ex == "jump_rope":
            return METS["jump_rope"]
    # Strength default based on RPE / exercise
    return pick_strength_met(item.get("rpe"), ex)


def estimate_kcal(item: Dict[str, Any], weight_kg_user: float) -> float:
    minutes = estimate_minutes(item)
    met = estimate_met(item)
    # Standard MET formula: kcal = MET * 3.5 * weight(kg) * minutes / 200
    kcal = met * 3.5 * float(weight_kg_user) * float(minutes) / 200.0
    return round(kcal, 1)


def compute_volume(item: Dict[str, Any]) -> float:
    # Strength volume proxy; for cardio, return 0; for bodyweight reps, assign small proxy
    weight = item.get("weight_kg")
    reps = item.get("reps")
    sets = item.get("sets")
    bp = item.get("body_part")
    if bp == "cardio":
        return 0.0
    if weight is None and reps and sets:
        return float(reps * sets) * 0.05  # proxy for bodyweight work
    if weight and reps and sets:
        return float(weight) * float(reps) * float(sets) / 1000.0
    return 0.0


def update_heat_with_items(heat_df: pd.DataFrame, items: List[Dict[str, Any]]) -> pd.DataFrame:
    today = dt.date.today()
    if heat_df.empty:
        row = {"date": today}
        for bp in BODY_PARTS:
            row[bp] = 0.0
        heat_df = pd.DataFrame([row])

    # Decay from last date to today, step by step (in case of gaps)
    last_date = heat_df["date"].max()
    if isinstance(last_date, pd.Timestamp):
        last_date = last_date.date()

    date_cursor = last_date
    current = heat_df[heat_df["date"] == last_date].iloc[0].to_dict()
    for bp in BODY_PARTS:
        current[bp] = float(current.get(bp, 0.0))

    while date_cursor < today:
        date_cursor = date_cursor + relativedelta(days=1)
        decayed = {bp: current[bp] * HEAT_DECAY for bp in BODY_PARTS}
        row = {"date": date_cursor}
        row.update(decayed)
        heat_df = pd.concat([heat_df, pd.DataFrame([row])], ignore_index=True)
        current = row

    # Add today's increments
    increments = {bp: 0.0 for bp in BODY_PARTS}
    for it in items:
        increments[it.get("body_part", "other")] += compute_volume(it)

    if today in set(heat_df["date"].tolist()):
        idx = heat_df.index[heat_df["date"] == today][0]
        for bp in BODY_PARTS:
            heat_df.loc[idx, bp] = float(heat_df.loc[idx, bp]) + increments[bp]
    else:
        row = {"date": today}
        for bp in BODY_PARTS:
            row[bp] = increments[bp]
        heat_df = pd.concat([heat_df, pd.DataFrame([row])], ignore_index=True)

    return heat_df


# ---------- Suggestion Generation ----------
SUGGESTION_SYS = (
    "You are a concise strength & conditioning assistant. Based on the user's profile and last 7 days, "
    "produce 3-5 actionable tips for today's training. Avoid medical claims. Use encouraging tone."
)

SUGGESTION_USER_TMPL = """
User profile: {profile}
Last 7 days summary:
  Total kcal: {kcal_7d}
  Volume (sum): {vol_7d}
  Heat (today): {heat_today}
Flags:
  Overload risk: {overload_flag}
  Undertrained parts: {under_parts}
  Rest-needed parts: {rest_parts}
Goal: {goal}
Please return 3-5 bullet points in Chinese, with 1 concrete mini-plan block for today.
"""


def generate_suggestions(profile: Dict[str, Any], df7: pd.DataFrame, heat_today: Dict[str, float]) -> str:
    # Simple flags
    kcal_7d = round(df7["est_kcal"].sum(), 1) if not df7.empty else 0.0
    vol_7d = round(df7["volume"].sum(), 2) if not df7.empty else 0.0

    # overload if last 3-day kcal > 7-day avg * 1.3
    kcal_3d = df7[df7["date"] >= (dt.date.today() - relativedelta(days=2))]["est_kcal"].sum() if not df7.empty else 0
    kcal_mean7 = df7["est_kcal"].mean() if not df7.empty else 0
    overload_flag = (kcal_3d > 3 * kcal_mean7 * 1.3) if kcal_mean7 else False

    under_parts = [bp for bp, v in heat_today.items() if v < HEAT_LOW]
    rest_parts = [bp for bp, v in heat_today.items() if v > HEAT_HIGH]

    if not OPENAI_AVAILABLE:
        # Simple fallback rules in Chinese
        bullets = []
        if overload_flag:
            bullets.append("最近训练负荷上升较快，建议今天适当降载或缩短时长，关注睡眠与补水。")
        if under_parts:
            bullets.append(f"以下部位近期训练不足：{', '.join(under_parts)}，今天可优先补练这些部位。")
        if rest_parts:
            bullets.append(f"以下部位近期负荷偏高：{', '.join(rest_parts)}，建议休息/轻量活动以恢复。")
        bullets.append("请控制主观强度在 RPE 7 左右，结束后进行 5–10 分钟拉伸与放松。")
        bullets.append("今日小计划：上肢推 3x5@RPE7；拉 3x8；核心 2x60s；有氧 15–20min 低强度。")
        return "\n- " + "\n- ".join(bullets)

    try:
        prof_compact = {
            "sex": profile.get("sex"),
            "age": profile.get("age"),
            "height_cm": profile.get("height_cm"),
            "weight_kg": profile.get("weight_kg"),
        }
        heat_today_compact = {k: round(v,2) for k,v in heat_today.items()}
        user_prompt = SUGGESTION_USER_TMPL.format(
            profile=json.dumps(prof_compact, ensure_ascii=False),
            kcal_7d=kcal_7d,
            vol_7d=vol_7d,
            heat_today=json.dumps(heat_today_compact, ensure_ascii=False),
            overload_flag=overload_flag,
            under_parts=under_parts,
            rest_parts=rest_parts,
            goal=profile.get("goal","fat_loss"),
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content": SUGGESTION_SYS},
                {"role":"user","content": user_prompt},
            ],
            temperature=0.4,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return "- 今天以 RPE 7 为主，优先练不足部位，过载部位降载或休息；训练后拉伸 5–10 分钟。\n- 小计划：上肢推 3x5；拉 3x8；核心 2x60s；低强度有氧 15–20min。"


# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI 健身日志助理 (MVP)", layout="wide")
st.title("AI 健身日志助理 · MVP")

# Sidebar: user profile
st.sidebar.header("基础信息 / Profile")
users_df = load_users()
user_row = users_df.iloc[0].to_dict() if not users_df.empty else {"height_cm":175,"weight_kg":75,"age":27,"sex":"male","goal":"fat_loss"}

height_cm = st.sidebar.number_input("身高 (cm)", value=float(user_row.get("height_cm",175.0)), step=1.0)
weight_kg = st.sidebar.number_input("体重 (kg)", value=float(user_row.get("weight_kg",75.0)), step=0.1)
age = st.sidebar.number_input("年龄", value=int(user_row.get("age",27)), step=1)
sex = st.sidebar.selectbox("性别", options=["male","female","other"], index=["male","female","other"].index(str(user_row.get("sex","male"))))
goal = st.sidebar.selectbox("目标", options=["fat_loss","hypertrophy","endurance"], index=["fat_loss","hypertrophy","endurance"].index(str(user_row.get("goal","fat_loss"))))

if st.sidebar.button("保存基础信息"):
    save_user_profile(height_cm, weight_kg, age, sex, goal)
    st.sidebar.success("已保存 ✅")

# Main: input area
st.subheader("输入今天的训练记录 · CN/EN 都可")
example = "卧推 60kg×5×3；硬拉 80kg×5×3；跑步 30 分钟；RPE 8"
log_text = st.text_area("示例：" + example, height=140)

col_a, col_b = st.columns([1,1])

with col_a:
    use_llm = st.checkbox("使用 OpenAI 抽取（无 Key 时自动回退正则）", value=True)

with col_b:
    if OPENAI_AVAILABLE:
        st.info("✅ 检测到 OPENAI_API_KEY，已启用 LLM 解析。")
    else:
        st.warning("未检测到 OPENAI_API_KEY，将使用正则兜底解析。")

if st.button("解析并保存"):
    today = dt.date.today()
    items_llm = llm_parse(log_text, today) if (use_llm and OPENAI_AVAILABLE and log_text.strip()) else []
    items_rx = regex_parse(log_text, today) if log_text.strip() else []

    # merge: prefer LLM items; if none found, use regex
    items = items_llm if items_llm else items_rx

    # enrich with kcal/volume
    enriched = []
    for it in items:
        it["body_part"] = map_body_part(it.get("exercise","")) if not it.get("body_part") else it["body_part"]
        kcal = estimate_kcal(it, weight_kg)
        vol = compute_volume(it)
        est_met = estimate_met(it)
        enriched.append({
            "date": it["date"],
            "exercise": to_snake(it.get("exercise","unknown")),
            "body_part": it.get("body_part","other"),
            "weight_kg": it.get("weight_kg"),
            "reps": it.get("reps"),
            "sets": it.get("sets"),
            "minutes": estimate_minutes(it),
            "rpe": it.get("rpe"),
            "est_met": est_met,
            "est_kcal": kcal,
            "volume": vol,
        })

    if not enriched:
        st.error("没有解析到有效条目，请调整输入格式或增加细节。")
    else:
        append_workouts(enriched)
        st.success(f"已保存 {len(enriched)} 条记录 ✅")
        st.dataframe(pd.DataFrame(enriched))

# Load data for charts
wdf = load_workouts()

if wdf.empty:
    st.info("暂无训练数据。请先输入并保存一次训练记录。")
    st.stop()

# Last 7 days slice
wdf["date"] = pd.to_datetime(wdf["date"]).dt.date
last7 = dt.date.today() - relativedelta(days=6)
w7 = wdf[wdf["date"] >= last7]

# Charts
st.subheader("趋势图")

# Calories trend (7d)
cal_7d = (
    w7.groupby("date", as_index=False)["est_kcal"].sum().rename(columns={"est_kcal":"kcal"})
)
cal_chart = alt.Chart(cal_7d).mark_line(point=True).encode(
    x=alt.X("date:T", title="日期"),
    y=alt.Y("kcal:Q", title="当日总消耗 (kcal)"),
    tooltip=["date:T","kcal:Q"],
).properties(height=260)

st.altair_chart(cal_chart, use_container_width=True)

# Optional: Volume trend
with st.expander("训练量趋势（可选）"):
    vol_7d = (
        w7.groupby("date", as_index=False)["volume"].sum()
    )
    vol_chart = alt.Chart(vol_7d).mark_line(point=True).encode(
        x=alt.X("date:T", title="日期"),
        y=alt.Y("volume:Q", title="当日训练量 (proxy)"),
        tooltip=["date:T","volume:Q"],
    ).properties(height=220)
    st.altair_chart(vol_chart, use_container_width=True)

# Body-part heat (decay + today increments from any new logs)
heat_df = load_heat()
heat_today_before = {} if heat_df.empty else {bp: float(heat_df.iloc[-1][bp]) for bp in BODY_PARTS}

# No new items at this moment; to ensure daily decay, call update with empty list
heat_df = update_heat_with_items(heat_df, [])

# Recompute today's increments from today's rows to reflect the view
today_rows = wdf[wdf["date"] == dt.date.today()]
increments = {bp: 0.0 for bp in BODY_PARTS}
for _, r in today_rows.iterrows():
    increments[str(r["body_part"]) ] += float(r["volume"]) if not pd.isna(r["volume"]) else 0.0

# Add increments on top of decayed base for charting (without double-saving)
heat_row_today = heat_df[heat_df["date"] == dt.date.today()]
if not heat_row_today.empty:
    base = heat_row_today.iloc[-1].to_dict()
    heat_today = {bp: float(base[bp]) + increments[bp] for bp in BODY_PARTS}
else:
    heat_today = {bp: increments[bp] for bp in BODY_PARTS}

heat_long = pd.DataFrame({"body_part": list(heat_today.keys()), "heat": list(heat_today.values())})

st.subheader("部位训练热度（带时间衰减）")
bar = alt.Chart(heat_long).mark_bar().encode(
    x=alt.X("body_part:N", title="部位"),
    y=alt.Y("heat:Q", title="热度 (proxy)", scale=alt.Scale(domain=[0, max(HEAT_HIGH*1.2, max(heat_long["heat"]) if not heat_long.empty else 3.5)])),
    tooltip=["body_part","heat"],
).properties(height=280)
rule_low = alt.Chart(pd.DataFrame({"y": [HEAT_LOW]})).mark_rule(strokeDash=[4,4]).encode(y="y")
rule_high = alt.Chart(pd.DataFrame({"y": [HEAT_HIGH]})).mark_rule(strokeDash=[4,4]).encode(y="y")

st.altair_chart(bar + rule_low + rule_high, use_container_width=True)

# Flags
need_more = [bp for bp, v in heat_today.items() if v < HEAT_LOW]
need_rest = [bp for bp, v in heat_today.items() if v > HEAT_HIGH]
cols = st.columns(2)
with cols[0]:
    if need_more:
        st.success("需要加强的部位：" + ", ".join(need_more))
    else:
        st.info("暂无需要加强的部位（低于阈值）。")
with cols[1]:
    if need_rest:
        st.warning("负荷偏高需休息的部位：" + ", ".join(need_rest))
    else:
        st.info("暂无需要休息的部位（高于阈值）。")

# Suggestions
st.subheader("一键生成今日建议")
if st.button("生成建议"):
    profile = {"height_cm": height_cm, "weight_kg": weight_kg, "age": age, "sex": sex, "goal": goal}
    df7 = w7.copy()
    sugg = generate_suggestions(profile, df7, heat_today)
    st.markdown(sugg)

# Footer
st.caption("MVP: 自然语言训练日志 → 结构化 → 卡路里估算 → 趋势图 → 部位热度（时间衰减）→ 一键建议。未来可对接设备数据以提升准确性。")
