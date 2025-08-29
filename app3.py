"""
AI Fitness Log Assistant (Streamlit) – MVP (English UI)
------------------------------------------------------
单文件 Streamlit 应用（**代码内注释为中文，界面/提示为英文**）：
  • 录入用户基础信息（身高/体重/年龄/性别）
  • 用 OpenAI API 将自然语言训练日志解析为结构化数据（**移除了正则兜底**，更简洁）
  • 估算当日卡路里（基于 MET 的近似公式）
  • 展示近 7 天卡路里趋势
  • 维护按部位的“热度”（带 7 天半衰期衰减），提示需要加强/需要休息
  • 一键生成英文建议（结合用户信息与近 7 天数据）
  • **支持自定义记录日期**（便于测试多天数据演示）

运行方式：
  1) pip install streamlit pandas altair python-dateutil openai
  2) 设置环境变量 OPENAI_API_KEY（必须）：
     - Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
  3) streamlit run app.py

说明：
  • 该版本移除了正则解析，完全依赖 LLM（更干净；你已确认可用 API Key）
  • 如果解析失败，会做一次自动重试（更稳健）
  • 训练量趋势已按你的要求移除
"""

import os
import json
import math
import datetime as dt
from typing import List, Dict, Any

import pandas as pd
import altair as alt
from dateutil.relativedelta import relativedelta
import streamlit as st

# =========================
# 配置区（中文注释，英文界面）
# =========================
DATA_DIR = "data"
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
WORKOUTS_CSV = os.path.join(DATA_DIR, "workouts.csv")
HEAT_CSV = os.path.join(DATA_DIR, "heat.csv")

BODY_PARTS = ["chest", "back", "legs", "shoulders", "arms", "core", "cardio", "other"]

# 动作→部位映射（用于兜底/规范化；LLM 也会给 body_part，这里做最终保证）
BODY_PART_MAP = {
    "bench_press": "chest", "incline_bench_press": "chest", "push_up": "chest",
    "overhead_press": "shoulders", "shoulder_press": "shoulders", "lateral_raise": "shoulders",
    "front_raise": "shoulders", "dumbbell_row": "back", "barbell_row": "back",
    "pull_up": "back", "chin_up": "back", "lat_pulldown": "back",
    "biceps_curl": "arms", "triceps_extension": "arms",
    "squat": "legs", "front_squat": "legs", "deadlift": "legs", "romanian_deadlift": "legs",
    "lunge": "legs", "leg_press": "legs", "plank": "core", "crunch": "core", "leg_raise": "core",
    "run": "cardio", "running": "cardio", "treadmill": "cardio", "cycle": "cardio",
    "cycling": "cardio", "bike": "cardio", "rower": "cardio", "rowing": "cardio", "jump_rope": "cardio",
}

# MET 估算（用于卡路里）：
METS = {
    "run_easy": 8.3, "run_hard": 11.0,
    "cycling_easy": 6.8, "cycling_hard": 8.0,
    "rowing": 7.0, "jump_rope": 10.0,
    "strength_medium": 3.5, "strength_hard": 6.0,
}

HALF_LIFE_DAYS = 7
HEAT_DECAY = math.exp(-math.log(2) / HALF_LIFE_DAYS)
HEAT_LOW = 1.0
HEAT_HIGH = 3.0

# OpenAI 客户端（必须）
import openai
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Please set your API key before running the app.")
openai.api_key = os.getenv("OPENAI_API_KEY")

# =========================
# 基础工具函数
# =========================

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(WORKOUTS_CSV):
        pd.DataFrame(columns=[
            "date","exercise","body_part","weight_kg","reps","sets","minutes","rpe","est_met","est_kcal","volume"
        ]).to_csv(WORKOUTS_CSV, index=False)
    if not os.path.exists(USERS_CSV):
        pd.DataFrame([{ "height_cm": 175, "weight_kg": 75, "age": 27, "sex": "male", "goal": "fat_loss" }]).to_csv(USERS_CSV, index=False)
    if not os.path.exists(HEAT_CSV):
        row = {"date": (dt.date.today() - relativedelta(days=1)).isoformat(), **{bp: 0.0 for bp in BODY_PARTS}}
        pd.DataFrame([row]).to_csv(HEAT_CSV, index=False)


def load_users() -> pd.DataFrame:
    ensure_dirs()
    return pd.read_csv(USERS_CSV)


def save_user_profile(height_cm: float, weight_kg: float, age: int, sex: str, goal: str):
    pd.DataFrame([{ "height_cm": height_cm, "weight_kg": weight_kg, "age": age, "sex": sex, "goal": goal }]).to_csv(USERS_CSV, index=False)


def load_workouts() -> pd.DataFrame:
    ensure_dirs()
    df = pd.read_csv(WORKOUTS_CSV)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        for c in ["weight_kg","reps","sets","minutes","rpe","est_met","est_kcal","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def append_workouts(rows: List[Dict[str, Any]]):
    if not rows: return
    df_old = load_workouts()
    df_new = pd.DataFrame(rows)
    pd.concat([df_old, df_new], ignore_index=True).to_csv(WORKOUTS_CSV, index=False)


def load_heat() -> pd.DataFrame:
    ensure_dirs()
    df = pd.read_csv(HEAT_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def save_heat(df: pd.DataFrame):
    df.to_csv(HEAT_CSV, index=False)

# =========================
# 解析与估算（全部走 LLM，中文注释）
# =========================
LLM_SYSTEM = (
    "You are a strict workout-log parser. Return ONLY valid JSON (UTF-8). "
    "Parse Chinese/English logs into list of items with fields: "
    "date (YYYY-MM-DD), exercise (snake_case), body_part (chest/back/legs/shoulders/arms/core/cardio/other), "
    "weight_kg (number|null), reps (int|null), sets (int|null), minutes (number|null), rpe (number|null). "
    "Infer body_part; default date = provided date if present, else today."
)

LLM_USER_TMPL = """
Text:
{log}
Use this date for missing dates: {date}
Output schema:
{{"items": [{{"date":"YYYY-MM-DD","exercise":"bench_press","body_part":"chest","weight_kg":60,"reps":5,"sets":3,"minutes":null,"rpe":8}}]}}
"""

# —— 统一化：转 snake_case ——
import re

def to_snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unknown"


def map_body_part_safe(exercise: str, body_part: str) -> str:
    # 以 LLM 的 body_part 为主；如果缺失/异常，用表映射兜底
    bp = (body_part or "").strip().lower()
    if bp in BODY_PARTS: return bp
    return BODY_PART_MAP.get(to_snake(exercise), "other")


def llm_parse(text: str, base_date: dt.date) -> List[Dict[str, Any]]:
    # 使用两次尝试：若 JSON 解析失败或空结果，做一次轻微改写后重试，提高稳健性
    def _call_llm(t: str) -> List[Dict[str, Any]]:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-",
            messages=[
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": LLM_USER_TMPL.format(log=t, date=base_date.isoformat())},
            ],
            temperature=0.0,
        )
        content = resp["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        items = data.get("items", [])
        # 规范化
        out = []
        for it in items:
            ex = to_snake(str(it.get("exercise", "")))
            bp = map_body_part_safe(ex, it.get("body_part"))
            d = it.get("date") or base_date.isoformat()
            out.append({
                "date": d,
                "exercise": ex,
                "body_part": bp,
                "weight_kg": it.get("weight_kg"),
                "reps": it.get("reps"),
                "sets": it.get("sets"),
                "minutes": it.get("minutes"),
                "rpe": it.get("rpe"),
            })
        return out

    try:
        items = _call_llm(text)
        if items: return items
    except Exception:
        pass
    # 轻微改写：去除奇怪符号、统一分隔符
    try:
        alt_text = text.replace("；", ";").replace("，", ",").replace("×", "x")
        items = _call_llm(alt_text)
        return items
    except Exception:
        return []

# —— 卡路里与热度 ——

def pick_strength_met(rpe: float, exercise: str) -> float:
    # RPE≥8 或常见复合大重量 → 6.0，否则 3.5
    hard_compounds = {"squat", "deadlift", "bench_press", "overhead_press", "barbell_row"}
    if (rpe or 0) >= 8.0 or to_snake(exercise) in hard_compounds:
        return METS["strength_hard"]
    return METS["strength_medium"]


def estimate_minutes(item: Dict[str, Any]) -> float:
    # 力量无时长时，估 2 分钟/组（含休息）
    m = item.get("minutes")
    if m is not None and m != "":
        try: return float(m) or 0.0
        except Exception: pass
    sets = float(item.get("sets") or 0)
    return round(2.0 * sets, 1)


def estimate_met(item: Dict[str, Any]) -> float:
    bp = item.get("body_part", "other")
    ex = to_snake(item.get("exercise", ""))
    if bp == "cardio" or ex in {"run","running","treadmill","cycle","cycling","bike","rower","rowing","jump_rope"}:
        if ex in {"running","run","treadmill"}: return METS["run_easy"]
        if ex in {"cycling","cycle","bike"}: return METS["cycling_easy"]
        if ex in {"rower","rowing"}: return METS["rowing"]
        if ex == "jump_rope": return METS["jump_rope"]
    return pick_strength_met(item.get("rpe"), ex)


def estimate_kcal(item: Dict[str, Any], weight_kg_user: float) -> float:
    # 标准 MET 公式：kcal = MET * 3.5 * 体重(kg) * 分钟 / 200
    minutes = estimate_minutes(item)
    met = estimate_met(item)
    return round(met * 3.5 * float(weight_kg_user) * float(minutes) / 200.0, 1)


def compute_volume(item: Dict[str, Any]) -> float:
    # 训练量（仅作部位热度的 proxy；cardio 记 0，自重按经验值）
    weight = item.get("weight_kg")
    reps = item.get("reps")
    sets = item.get("sets")
    bp = item.get("body_part")
    if bp == "cardio": return 0.0
    if weight and reps and sets:
        return float(weight) * float(reps) * float(sets) / 1000.0
    if (not weight) and reps and sets:
        return float(reps * sets) * 0.05
    return 0.0


def update_heat_with_items(heat_df: pd.DataFrame, items: List[Dict[str, Any]]) -> pd.DataFrame:
    # 将热度衰减推进到“items 最晚日期”或“今天”，然后叠加当天增量并保存
    target_date = max([dt.date.fromisoformat(i["date"]) for i in items], default=dt.date.today())
    if heat_df.empty:
        heat_df = pd.DataFrame([{"date": dt.date.today(), **{bp: 0.0 for bp in BODY_PARTS}}])

    last_date = heat_df["date"].max()
    if isinstance(last_date, pd.Timestamp):
        last_date = last_date.date()

    current = heat_df[heat_df["date"] == last_date].iloc[0].to_dict()
    for bp in BODY_PARTS: current[bp] = float(current.get(bp, 0.0))

    # 逐日衰减到目标日期
    while last_date < target_date:
        last_date = last_date + relativedelta(days=1)
        decayed = {bp: current[bp] * HEAT_DECAY for bp in BODY_PARTS}
        row = {"date": last_date, **decayed}
        heat_df = pd.concat([heat_df, pd.DataFrame([row])], ignore_index=True)
        current = row

    # 叠加目标日期的增量（从 items 聚合）
    increments = {bp: 0.0 for bp in BODY_PARTS}
    for it in items:
        increments[it.get("body_part", "other")] += compute_volume(it)

    idx = heat_df.index[heat_df["date"] == target_date][0]
    for bp in BODY_PARTS:
        heat_df.loc[idx, bp] = float(heat_df.loc[idx, bp]) + increments[bp]

    save_heat(heat_df)
    return heat_df

# =========================
# Streamlit UI（英文界面）
# =========================
st.set_page_config(page_title="AI Fitness Log Assistant (MVP)", layout="wide")
st.title("AI Fitness Log Assistant · MVP")

# Sidebar – Profile
st.sidebar.header("Profile")
users_df = load_users()
user_row = users_df.iloc[0].to_dict() if not users_df.empty else {"height_cm":175,"weight_kg":75,"age":27,"sex":"male","goal":"fat_loss"}

height_cm = st.sidebar.number_input("Height (cm)", value=float(user_row.get("height_cm",175.0)), step=1.0)
weight_kg = st.sidebar.number_input("Weight (kg)", value=float(user_row.get("weight_kg",75.0)), step=0.1)
age = st.sidebar.number_input("Age", value=int(user_row.get("age",27)), step=1)
sex = st.sidebar.selectbox("Sex", options=["male","female","other"], index=["male","female","other"].index(str(user_row.get("sex","male"))))
goal = st.sidebar.selectbox("Goal", options=["fat_loss","hypertrophy","endurance"], index=["fat_loss","hypertrophy","endurance"].index(str(user_row.get("goal","fat_loss"))))

if st.sidebar.button("Save Profile"):
    save_user_profile(height_cm, weight_kg, age, sex, goal)
    st.sidebar.success("Saved ✅")

# Main – Input
st.subheader("Add a training log (English or Chinese)")
example = "Bench press 60kg x5 x3; Deadlift 80kg x5 x3; Running 30 minutes; RPE 8"
log_text = st.text_area("Example: " + example, height=140)

# 选择“记录日期”（用于测试多天数据）
record_date = st.date_input("Record date", value=dt.date.today())

if st.button("Parse & Save"):
    if not log_text.strip():
        st.error("Please enter your training log text.")
    else:
        items = llm_parse(log_text, record_date)
        if not items:
            st.error("No valid items parsed. Please try adjusting the text.")
        else:
            enriched = []
            for it in items:
                it["body_part"] = map_body_part_safe(it.get("exercise",""), it.get("body_part"))
                kcal = round(float(estimate_kcal(it, weight_kg)), 1)
                vol = compute_volume(it)
                est_met = estimate_met(it)
                enriched.append({
                    "date": it["date"],
                    "exercise": it.get("exercise","unknown"),
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
            append_workouts(enriched)
            # 同步更新热度并保存（确保部位图立刻反映）
            heat_df_current = load_heat()
            update_heat_with_items(heat_df_current, enriched)
            st.success(f"Saved {len(enriched)} items for {record_date.isoformat()} ✅")
            st.dataframe(pd.DataFrame(enriched))

# Load data for charts
wdf = load_workouts()
if wdf.empty:
    st.info("No data yet. Please add your first log.")
    st.stop()

# Last 7 days calories
st.subheader("7-day calories trend")
wdf["date"] = pd.to_datetime(wdf["date"]).dt.date
last7 = dt.date.today() - relativedelta(days=6)
w7 = wdf[wdf["date"] >= last7]
cal_7d = w7.groupby("date", as_index=False)["est_kcal"].sum().rename(columns={"est_kcal":"kcal"})
cal_chart = alt.Chart(cal_7d).mark_line(point=True).encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("kcal:Q", title="Daily total (kcal)"),
    tooltip=["date:T","kcal:Q"],
).properties(height=260)
st.altair_chart(cal_chart, use_container_width=True)

# Body-part heat (decay already applied & saved on each write)
st.subheader("Body-part training heat (with decay)")
heat_df = load_heat()
if not heat_df.empty:
    today_row = heat_df.iloc[-1].to_dict()
    heat_today = {bp: float(today_row[bp]) for bp in BODY_PARTS}
    heat_long = pd.DataFrame({"body_part": list(heat_today.keys()), "heat": list(heat_today.values())})
    bar = alt.Chart(heat_long).mark_bar().encode(
        x=alt.X("body_part:N", title="Body part"),
        y=alt.Y("heat:Q", title="Heat (proxy)", scale=alt.Scale(domain=[0, max(HEAT_HIGH*1.2, max(heat_long["heat"]) if not heat_long.empty else 3.5)])),
        tooltip=["body_part","heat"],
    ).properties(height=280)
    rule_low = alt.Chart(pd.DataFrame({"y": [HEAT_LOW]})).mark_rule(strokeDash=[4,4]).encode(y="y")
    rule_high = alt.Chart(pd.DataFrame({"y": [HEAT_HIGH]})).mark_rule(strokeDash=[4,4]).encode(y="y")
    st.altair_chart(bar + rule_low + rule_high, use_container_width=True)

    need_more = [bp for bp, v in heat_today.items() if v < HEAT_LOW]
    need_rest = [bp for bp, v in heat_today.items() if v > HEAT_HIGH]
    cols = st.columns(2)
    with cols[0]:
        st.info("Needs more: " + (", ".join(need_more) if need_more else "None"))
    with cols[1]:
        st.warning("Needs rest: " + (", ".join(need_rest) if need_rest else "None"))
else:
    st.info("No heat data yet. Add a log to initialize.")

# Suggestions (English)
SUGGESTION_SYS = (
    "You are a concise strength & conditioning assistant. Based on the user's profile and last 7 days, "
    "produce 3-5 actionable tips for today's training. Avoid medical claims. Encourage, short bullets. Include one mini-plan."
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
Return 3-5 bullet points in ENGLISH, plus one short mini-plan block for TODAY.
"""

def generate_suggestions(profile: Dict[str, Any], df7: pd.DataFrame, heat_today: Dict[str, float]) -> str:
    kcal_7d = round(df7["est_kcal"].sum(), 1) if not df7.empty else 0.0
    vol_7d = round(df7["volume"].sum(), 2) if not df7.empty else 0.0
    kcal_3d = df7[df7["date"] >= (dt.date.today() - relativedelta(days=2))]["est_kcal"].sum() if not df7.empty else 0
    kcal_mean7 = df7["est_kcal"].mean() if not df7.empty else 0
    overload_flag = (kcal_3d > 3 * kcal_mean7 * 1.3) if kcal_mean7 else False

    under_parts = [bp for bp, v in heat_today.items() if v < HEAT_LOW]
    rest_parts = [bp for bp, v in heat_today.items() if v > HEAT_HIGH]

    prof_compact = {k: profile.get(k) for k in ["sex","age","height_cm","weight_kg"]}
    heat_today_compact = {k: round(v,2) for k,v in heat_today.items()}
    prompt = SUGGESTION_USER_TMPL.format(
        profile=json.dumps(prof_compact, ensure_ascii=False),
        kcal_7d=kcal_7d, vol_7d=vol_7d,
        heat_today=json.dumps(heat_today_compact, ensure_ascii=False),
        overload_flag=overload_flag,
        under_parts=under_parts,
        rest_parts=rest_parts,
        goal=profile.get("goal","fat_loss"),
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content": SUGGESTION_SYS},{"role":"user","content": prompt}],
            temperature=0.4,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return (
            "- Keep RPE around 7 and monitor recovery."
            "- Prioritize undertrained parts; deload/rest overtrained parts."
            "- Hydration and 5–10 min cool-down."
            "Mini-plan: Push 3x5 @RPE7; Pull 3x8; Core 2x60s; 15–20min easy cardio."
        )

st.subheader("Generate today's suggestions")
if st.button("Generate suggestions"):
    profile = {"height_cm": height_cm, "weight_kg": weight_kg, "age": age, "sex": sex, "goal": goal}
    df7 = w7.copy()
    today_heat_row = load_heat().iloc[-1].to_dict() if not load_heat().empty else {bp:0.0 for bp in BODY_PARTS}
    heat_today = {bp: float(today_heat_row[bp]) for bp in BODY_PARTS}
    st.markdown(generate_suggestions(profile, df7, heat_today))

st.caption("MVP: free-form → structured via OpenAI → calories → 7-day trend → body-part heat with decay → English suggestions. Date selector helps demo multiple days.")
