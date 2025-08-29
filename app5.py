"""
AI Fitness Log Assistant (Streamlit) – Stable Minimal (English UI)
------------------------------------------------------------------
界面/提示全英文；代码注释中文。重点：更简单、更稳；带 Debug 面板；支持选日期测试多天；
可一键重置数据，便于演示。
"""

import os, re, json, math, time, datetime as dt
from typing import List, Dict, Any, Tuple

import pandas as pd
import altair as alt
from dateutil.relativedelta import relativedelta
import streamlit as st

# =========================
# 基础配置（中文注释，英文界面）
# =========================
DATA_DIR = "data"
USERS_CSV = os.path.join(DATA_DIR, "users.csv")
WORKOUTS_CSV = os.path.join(DATA_DIR, "workouts.csv")
HEAT_CSV = os.path.join(DATA_DIR, "heat.csv")

BODY_PARTS = ["chest", "back", "legs", "shoulders", "arms", "core", "cardio", "other"]
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

# MET 估算（用于卡路里）
METS = {
    "run_easy": 8.3, "run_hard": 11.0,
    "cycling_easy": 6.8, "cycling_hard": 8.0,
    "rowing": 7.0, "jump_rope": 10.0,
    "strength_medium": 3.5, "strength_hard": 6.0,
}

HALF_LIFE_DAYS = 7
HEAT_DECAY = math.exp(-math.log(2) / HALF_LIFE_DAYS)
HEAT_LOW, HEAT_HIGH = 1.0, 3.0

# OpenAI（必须）
import openai
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Please set it before running the app.")
openai.api_key = os.getenv("OPENAI_API_KEY")

# =========================
# 文件 I/O
# =========================
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(WORKOUTS_CSV):
        pd.DataFrame(columns=[
            "date","exercise","body_part","weight_kg","reps","sets","minutes","rpe","est_met","est_kcal","volume"
        ]).to_csv(WORKOUTS_CSV, index=False)
    if not os.path.exists(USERS_CSV):
        pd.DataFrame([{"height_cm":175,"weight_kg":75,"age":27,"sex":"male","goal":"fat_loss"}]).to_csv(USERS_CSV, index=False)
    if not os.path.exists(HEAT_CSV):
        row = {"date": (dt.date.today() - relativedelta(days=1)).isoformat(), **{bp:0.0 for bp in BODY_PARTS}}
        pd.DataFrame([row]).to_csv(HEAT_CSV, index=False)

def load_users() -> pd.DataFrame:
    ensure_dirs(); return pd.read_csv(USERS_CSV)

def save_user_profile(height_cm: float, weight_kg: float, age: int, sex: str, goal: str):
    pd.DataFrame([{"height_cm":height_cm,"weight_kg":weight_kg,"age":age,"sex":sex,"goal":goal}]).to_csv(USERS_CSV, index=False)

def load_workouts() -> pd.DataFrame:
    ensure_dirs(); df = pd.read_csv(WORKOUTS_CSV)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        for c in ["weight_kg","reps","sets","minutes","rpe","est_met","est_kcal","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def append_workouts(rows: List[Dict[str, Any]]):
    if not rows: return
    df = pd.concat([load_workouts(), pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(WORKOUTS_CSV, index=False)

def load_heat() -> pd.DataFrame:
    ensure_dirs(); df = pd.read_csv(HEAT_CSV)
    if not df.empty: df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def save_heat(df: pd.DataFrame):
    df.to_csv(HEAT_CSV, index=False)

# =========================
# 解析（LLM）+ Debug
# =========================
LLM_SYSTEM = (
    "You are a strict workout-log parser. Return ONLY valid JSON (UTF-8). "
    "Parse Chinese/English logs into list of items with fields: "
    "date (YYYY-MM-DD), exercise (snake_case), body_part (chest/back/legs/shoulders/arms/core/cardio/other), "
    "weight_kg (number|null), reps (int|null), sets (int|null), minutes (number|null), rpe (number|null). "
    "Infer body_part; default date = provided date if present, else the base date."
)

LLM_USER_TMPL = """
Text:
{log}
Base date for missing dates: {date}
Return EXACTLY one JSON object with key `items` only. No prose, no code fences.
Output schema:
{{"items": [{{"date":"YYYY-MM-DD","exercise":"bench_press","body_part":"chest","weight_kg":60,"reps":5,"sets":3,"minutes":null,"rpe":8}}]}}
"""


JSON_BLOCK = re.compile(r"\{[\s\S]*\}")  # 从回复里保险抓 JSON 块

def to_snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unknown"

def call_llm_parse(text: str, base_date: dt.date, model: str, retries: int = 1, timeout_s: int = 25) -> Tuple[str, Dict[str, Any]]:
    """返回 (raw_content, parsed_json)。失败时 parsed_json = {}，raw_content给出原始错误/文本"""
    last_raw = ""
    for attempt in range(retries + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=model,  # "gpt-4o-mini" 或 "gpt-4o"
                messages=[
                    {"role": "system", "content": LLM_SYSTEM},
                    {"role": "user", "content": LLM_USER_TMPL.format(log=text, date=base_date.isoformat())},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},  # 尽量让它只回 JSON
                timeout=timeout_s,
            )
            last_raw = resp["choices"][0]["message"]["content"].strip()

            # 调用 LLM 后
            content = resp["choices"][0]["message"]["content"].strip()
            st.session_state["debug"]["raw_response"] = content

            # 页面底部放一个 expander
            with st.expander("Debug panel (copy/paste to share)", expanded=False):
                st.code(st.session_state["debug"].get("prompt_user", "<no prompt>"), language="text")
                st.code(st.session_state["debug"].get("raw_response", "<no raw>"), language="json")
            ###

            # 直接 JSON
            try:
                return last_raw, json.loads(last_raw)
            except Exception:
                m = JSON_BLOCK.search(last_raw)
                if m:
                    try:
                        return last_raw, json.loads(m.group(0))
                    except Exception:
                        pass
            # 小清洗后再试一次
            text = text.replace("；", ";").replace("，", ",").replace("×", "x")
        except Exception as e:
            last_raw = f"<error: {e}>"
        time.sleep(0.2)
    return last_raw, {}

def normalize_items(parsed: Dict[str, Any], base_date: dt.date) -> List[Dict[str, Any]]:
    items = parsed.get("items") or []
    out = []
    for it in items:
        ex = to_snake(str(it.get("exercise", "")))
        bp = (it.get("body_part") or "").strip().lower()
        bp = bp if bp in BODY_PARTS else BODY_PART_MAP.get(ex, "other")
        d = it.get("date") or base_date.isoformat()
        out.append({
            "date": d, "exercise": ex, "body_part": bp,
            "weight_kg": it.get("weight_kg"), "reps": it.get("reps"),
            "sets": it.get("sets"), "minutes": it.get("minutes"), "rpe": it.get("rpe"),
        })
    return out

# =========================
# 估算（卡路里/热度）
# =========================
def pick_strength_met(rpe: float, exercise: str) -> float:
    hard = {"squat","deadlift","bench_press","overhead_press","barbell_row"}
    if (rpe or 0) >= 8.0 or to_snake(exercise) in hard: return METS["strength_hard"]
    return METS["strength_medium"]

def estimate_minutes(item: Dict[str, Any]) -> float:
    m = item.get("minutes")
    if m not in (None, ""):
        try: return float(m) or 0.0
        except Exception: pass
    sets = float(item.get("sets") or 0)
    return round(2.0 * sets, 1)  # 力量默认 2 分钟/组

def estimate_met(item: Dict[str, Any]) -> float:
    bp = item.get("body_part","other"); ex = to_snake(item.get("exercise",""))
    if bp == "cardio" or ex in {"run","running","treadmill","cycle","cycling","bike","rower","rowing","jump_rope"}:
        if ex in {"running","run","treadmill"}: return METS["run_easy"]
        if ex in {"cycling","cycle","bike"}: return METS["cycling_easy"]
        if ex in {"rower","rowing"}: return METS["rowing"]
        if ex == "jump_rope": return METS["jump_rope"]
    return pick_strength_met(item.get("rpe"), ex)

def estimate_kcal(item: Dict[str, Any], weight_kg_user: float) -> float:
    # kcal = MET * 3.5 * 体重(kg) * 分钟 / 200
    minutes = estimate_minutes(item); met = estimate_met(item)
    return round(met * 3.5 * float(weight_kg_user) * float(minutes) / 200.0, 1)

def compute_volume(item: Dict[str, Any]) -> float:
    w, r, s, bp = item.get("weight_kg"), item.get("reps"), item.get("sets"), item.get("body_part")
    if bp == "cardio": return 0.0
    if w and r and s: return float(w) * float(r) * float(s) / 1000.0
    if (not w) and r and s: return float(r * s) * 0.05  # 自重粗略量化
    return 0.0

def update_heat_with_items(heat_df: pd.DataFrame, items: List[Dict[str, Any]]) -> pd.DataFrame:
    target_date = max([dt.date.fromisoformat(i["date"]) for i in items], default=dt.date.today())
    if heat_df.empty:
        heat_df = pd.DataFrame([{"date": dt.date.today(), **{bp:0.0 for bp in BODY_PARTS}}])
    heat_df = heat_df.sort_values("date")
    last_date = heat_df["date"].iloc[-1]
    if isinstance(last_date, pd.Timestamp): last_date = last_date.date()
    current = heat_df[heat_df["date"] == last_date].iloc[0].to_dict()
    for bp in BODY_PARTS: current[bp] = float(current.get(bp, 0.0))
    # 衰减推进到目标日期
    while last_date < target_date:
        last_date = last_date + relativedelta(days=1)
        decayed = {bp: current[bp] * HEAT_DECAY for bp in BODY_PARTS}
        row = {"date": last_date, **decayed}
        heat_df = pd.concat([heat_df, pd.DataFrame([row])], ignore_index=True)
        current = row
    # 叠加增量
    inc = {bp: 0.0 for bp in BODY_PARTS}
    for it in items: inc[it.get("body_part","other")] += compute_volume(it)
    idx = heat_df.index[heat_df["date"] == target_date][0]
    for bp in BODY_PARTS: heat_df.loc[idx, bp] = float(heat_df.loc[idx, bp]) + inc[bp]
    save_heat(heat_df); return heat_df

# =========================
# UI（英文）+ Debug 面板
# =========================
st.set_page_config(page_title="AI Fitness Log Assistant (Stable Minimal)", layout="wide")
st.title("AI Fitness Log Assistant · Stable Minimal")

# Sidebar: profile & settings
st.sidebar.header("Profile")
u = load_users().iloc[0].to_dict() if not load_users().empty else {"height_cm":175,"weight_kg":75,"age":27,"sex":"male","goal":"fat_loss"}
height_cm = st.sidebar.number_input("Height (cm)", value=float(u.get("height_cm",175)), step=1.0)
weight_kg = st.sidebar.number_input("Weight (kg)", value=float(u.get("weight_kg",75)), step=0.1)
age = st.sidebar.number_input("Age", value=int(u.get("age",27)), step=1)
sex = st.sidebar.selectbox("Sex", ["male","female","other"], index=["male","female","other"].index(str(u.get("sex","male"))))
goal = st.sidebar.selectbox("Goal", ["fat_loss","hypertrophy","endurance"], index=["fat_loss","hypertrophy","endurance"].index(str(u.get("goal","fat_loss"))))
if st.sidebar.button("Save Profile"):
    save_user_profile(height_cm, weight_kg, age, sex, goal); st.sidebar.success("Saved ✅")

st.sidebar.header("Settings")
model = st.sidebar.selectbox("Model", ["gpt-4o-mini","gpt-4o"], index=0)
retries = st.sidebar.slider("Retries", 0, 3, 1)
timeout_s = st.sidebar.slider("Timeout (s)", 5, 60, 25)

with st.sidebar.expander("Data ops"):
    if st.button("Reset workouts.csv"):
        if os.path.exists(WORKOUTS_CSV): os.remove(WORKOUTS_CSV)
        load_workouts(); st.success("workouts.csv reset")
    if st.button("Reset heat.csv"):
        if os.path.exists(HEAT_CSV): os.remove(HEAT_CSV)
        load_heat(); st.success("heat.csv reset")

# Main input
st.subheader("Add a training log (EN or CN)")
c1,c2,c3 = st.columns(3)
with c1:
    if st.button("Example • Strength + Cardio"):
        st.session_state["example_text"] = "Bench press 60kg x5 x3; Deadlift 80kg x5 x3; Running 30 minutes; RPE 8"
with c2:
    if st.button("Example • Chinese"):
        st.session_state["example_text"] = "卧推 60kg×5×3；硬拉 80kg×5×3；跑步 30 分钟；RPE 8"
with c3:
    if st.button("Example • Bodyweight"):
        st.session_state["example_text"] = "Push-up 20 reps x3; Plank 2 minutes; Cycling 25 minutes; RPE 6"
example_text = st.session_state.get("example_text", "Bench press 60kg x5 x3; Deadlift 80kg x5 x3; Running 30 minutes; RPE 8")
log_text = st.text_area("Paste your log here:", value=example_text, height=140)

record_date = st.date_input("Record date", value=dt.date.today())

# Debug store

# 在 llm_parse 调用前保存 prompt，在拿到回复后保存原始内容
if "debug" not in st.session_state: st.session_state["debug"] = {}

prompt_user = LLM_USER_TMPL.format(log=log_text, date=record_date.isoformat())
st.session_state["debug"]["prompt_user"] = prompt_user


if st.button("Parse & Save"):
    if not log_text.strip():
        st.error("Please enter your training log text.")
    else:
        raw, parsed = call_llm_parse(log_text, record_date, model=model, retries=retries, timeout_s=timeout_s)
        st.session_state["debug"] = {
            "prompt_user": LLM_USER_TMPL.format(log=log_text, date=record_date.isoformat()),
            "raw_response": raw,
            "parsed_json": parsed,
        }
        items = normalize_items(parsed, record_date)
        if not items:
            st.error("No valid items parsed. Please try adjusting the text.")
        else:
            enriched = []
            for it in items:
                kcal = estimate_kcal(it, weight_kg)
                vol = compute_volume(it)
                est_met = estimate_met(it)
                enriched.append({
                    "date": it["date"], "exercise": it.get("exercise","unknown"),
                    "body_part": it.get("body_part","other"),
                    "weight_kg": it.get("weight_kg"), "reps": it.get("reps"),
                    "sets": it.get("sets"), "minutes": estimate_minutes(it), "rpe": it.get("rpe"),
                    "est_met": est_met, "est_kcal": kcal, "volume": vol,
                })
            append_workouts(enriched)
            update_heat_with_items(load_heat(), enriched)  # 立刻更新热度
            st.success(f"Saved {len(enriched)} items for {record_date.isoformat()} ✅")
            st.dataframe(pd.DataFrame(enriched))



# Debug 面板（请把三段内容复制给我方便定位）
with st.expander("Debug panel (copy/paste to share)", expanded=False):
    dbg = st.session_state["debug"]
    st.code(dbg.get("prompt_user","<no prompt>"), language="text")
    st.code(dbg.get("raw_response","<no raw response>"), language="json")
    st.code(json.dumps(dbg.get("parsed_json",{}), ensure_ascii=False, indent=2), language="json")

# Charts
wdf = load_workouts()
if wdf.empty:
    st.info("No data yet. Please add your first log.")
    st.stop()

st.subheader("7-day calories trend")
wdf["date"] = pd.to_datetime(wdf["date"]).dt.date
last7 = dt.date.today() - relativedelta(days=6)
w7 = wdf[wdf["date"] >= last7]
cal_7d = w7.groupby("date", as_index=False)["est_kcal"].sum().rename(columns={"est_kcal":"kcal"})
st.altair_chart(
    alt.Chart(cal_7d).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("kcal:Q", title="Daily total (kcal)"),
        tooltip=["date:T","kcal:Q"],
    ).properties(height=260),
    use_container_width=True
)

st.subheader("Body-part training heat (with decay)")
hdf = load_heat()
if not hdf.empty:
    hdf = hdf.sort_values("date")
    #today_row = hdf.iloc[-1].to_dict()
    heat_df = hdf.iloc.sort_values("date")
    today_row = heat_df.iloc[-1].to_dict()

    heat_today = {bp: float(today_row[bp]) for bp in BODY_PARTS}
    heat_long = pd.DataFrame({"body_part": list(heat_today.keys()), "heat": list(heat_today.values())})
    bar = alt.Chart(heat_long).mark_bar().encode(
        x=alt.X("body_part:N", title="Body part"),
        y=alt.Y("heat:Q", title="Heat (proxy)",
                scale=alt.Scale(domain=[0, max(HEAT_HIGH*1.2, max(heat_long["heat"]) if not heat_long.empty else 3.5)])),
        tooltip=["body_part","heat"],
    ).properties(height=280)
    rule_low  = alt.Chart(pd.DataFrame({"y":[HEAT_LOW]})).mark_rule(strokeDash=[4,4]).encode(y="y")
    rule_high = alt.Chart(pd.DataFrame({"y":[HEAT_HIGH]})).mark_rule(strokeDash=[4,4]).encode(y="y")
    st.altair_chart(bar + rule_low + rule_high, use_container_width=True)

    need_more = [bp for bp, v in heat_today.items() if v < HEAT_LOW]
    need_rest = [bp for bp, v in heat_today.items() if v > HEAT_HIGH]
    c1, c2 = st.columns(2)
    with c1: st.info("Needs more: " + (", ".join(need_more) if need_more else "None"))
    with c2: st.warning("Needs rest: " + (", ".join(need_rest) if need_rest else "None"))
else:
    st.info("No heat data yet. Add a log to initialize.")

# Suggestions（延用原思路）
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
def generate_suggestions(profile: Dict[str, Any], df7: pd.DataFrame, heat_today: Dict[str, float], model: str) -> str:
    kcal_7d = round(df7["est_kcal"].sum(), 1) if not df7.empty else 0.0
    vol_7d = round(df7["volume"].sum(), 2) if not df7.empty else 0.0
    kcal_3d = df7[df7["date"] >= (dt.date.today() - relativedelta(days=2))]["est_kcal"].sum() if not df7.empty else 0
    kcal_mean7 = df7["est_kcal"].mean() if not df7.empty else 0
    overload_flag = (kcal_3d > 3 * kcal_mean7 * 1.3) if kcal_mean7 else False
    under_parts = [bp for bp, v in heat_today.items() if v < HEAT_LOW]
    rest_parts = [bp for bp, v in heat_today.items() if v > HEAT_HIGH]

    prof = {k: profile.get(k) for k in ["sex","age","height_cm","weight_kg"]}
    heat_today_compact = {k: round(v,2) for k,v in heat_today.items()}
    prompt = SUGGESTION_USER_TMPL.format(
        profile=json.dumps(prof, ensure_ascii=False),
        kcal_7d=kcal_7d, vol_7d=vol_7d,
        heat_today=json.dumps(heat_today_compact, ensure_ascii=False),
        overload_flag=overload_flag, under_parts=under_parts, rest_parts=rest_parts,
        goal=profile.get("goal","fat_loss"),
    )
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content": SUGGESTION_SYS},
                      {"role":"user","content": prompt}],
            temperature=0.4, timeout=25,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return (
            "- Keep RPE around 7 and monitor recovery.\n"
            "- Prioritize undertrained parts; deload/rest overtrained parts.\n"
            "- Hydration and 5–10 min cool-down.\n"
            "Mini-plan: Push 3x5 @RPE7; Pull 3x8; Core 2x60s; 15–20min easy cardio."
        )

st.subheader("Generate today's suggestions")
if st.button("Generate suggestions"):
    profile = {"height_cm": height_cm, "weight_kg": weight_kg, "age": age, "sex": sex, "goal": goal}
    hdf = load_heat(); hdf = hdf.sort_values("date") if not hdf.empty else hdf
    row = hdf.iloc[-1].to_dict() if not hdf.empty else {bp:0.0 for bp in BODY_PARTS}
    heat_today = {bp: float(row[bp]) for bp in BODY_PARTS}
    st.markdown(generate_suggestions(profile, w7.copy(), heat_today, model=model))

st.caption("Stable Minimal: strict JSON parse + visible debug + date selector + immediate heat update. Send me the Debug panel outputs if anything looks off.")
