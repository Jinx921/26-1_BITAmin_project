import streamlit as st
import pandas as pd
import joblib, json
import numpy as np
import io
from pathlib import Path

st.set_page_config(
    page_title="서울시 따릉이 고장 예측",
    page_icon="🚲",
    layout="wide"
)

# CSS
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"], .stApp {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 1.2rem !important;
    max-width: 1280px !important;
}
header[data-testid="stHeader"] {background: transparent;}
.title-wrap { background: transparent; color: #0f172a; padding: 0; border-radius: 0; margin-bottom: 2px; }
.title-main {
    margin: 0;
    font-size: 2.0rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.2px;
    color: #0f172a;
    padding-top: 4px;
}
.small-muted { color:#6b7280; font-size:1.0rem; }
.legend-wrap { display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 8px 0; }
.badge {
    padding: 5px 10px;
    border-radius: 999px;
    color: white;
    font-weight: 700;
    font-size: 0.86rem;
    display: inline-block;
}
.badge-danger { background:#d62828; }
.badge-warn { background:#f77f00; }
.badge-safe { background:#2a9d8f; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg,#f8fafc,#e9eef5) !important;
    border: 1px solid #dbe3eb !important;
    border-radius: 12px !important;
    padding: 12px !important;
    box-shadow: none !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #6b7280 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
}
.stDownloadButton button {
    background: #0f172a !important;
    color: white !important;
    border: 1px solid #0f172a !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.stDownloadButton button:hover {
    background: #1e293b !important;
}
div[data-testid="stHorizontalBlock"] { gap: 14px !important; }

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
}
.section-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.section-icon-blue { background: #eff6ff; }
.section-icon-violet { background: #f5f3ff; }
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
    line-height: 1.3;
}
.section-desc {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 2px 0 0 0;
}

/* ── Table Header ── */
.tbl-head {
    padding: 13px 16px;
    text-align: left;
    font-size: 12.5px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #475569;
}
.tbl-head-center {
    padding: 13px 16px;
    text-align: center;
    font-size: 12.5px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #475569;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  데이터 / 모델 (경로만 프로젝트 구조에 맞게 조정)
# ══════════════════════════════════════════════
project_root = Path(__file__).resolve().parent.parent
base = project_root / "processed_data" / "heeseo"
if not base.exists():
    # fallback: 파일이 페이지 파일과 같은 폴더에 있을 때
    base = Path(__file__).parent
@st.cache_data
def load_raw_data():
    rental_daily = pd.read_pickle(base / "rental_daily.pkl")
    BR = pd.read_pickle(base / "BR.pkl")
    rental_daily["date"] = pd.to_datetime(rental_daily["date"])
    BR["date"] = pd.to_datetime(BR["date"])
    return rental_daily, BR

def build_snapshot_daily(rental_daily, BR, snapshot_date, window_days=30):
    snapshot_date = pd.Timestamp(snapshot_date)
    future_end = snapshot_date + pd.Timedelta(days=window_days)
    rental_past = rental_daily[rental_daily["date"] <= snapshot_date].copy()
    fault_past  = BR[BR["date"] <= snapshot_date].copy()
    if rental_past.empty:
        return pd.DataFrame()
    rental_agg = (
        rental_past.groupby("자전거번호")
        .agg(총이용거리=("일일이용거리","sum"), 총대여횟수=("일일대여횟수","sum"), 마지막대여일=("date","max"))
        .reset_index()
    )
    def recent_agg(days):
        start = snapshot_date - pd.Timedelta(days=days)
        temp  = rental_past[rental_past["date"] > start]
        return (
            temp.groupby("자전거번호")
            .agg(**{f"최근{days}일이용거리":("일일이용거리","sum"), f"최근{days}일대여횟수":("일일대여횟수","sum")})
            .reset_index()
        )
    r7, r30, r90 = recent_agg(7), recent_agg(30), recent_agg(90)
    if fault_past.empty:
        fault_agg = pd.DataFrame(columns=["자전거번호","총고장횟수","마지막고장일","주요고장유형"])
    else:
        fault_type_mode = (
            fault_past.groupby("bike")["type"]
            .agg(lambda x: x.mode().iat[0] if not x.mode().empty else np.nan)
            .rename("주요고장유형").reset_index().rename(columns={"bike":"자전거번호"})
        )
        fault_agg = (
            fault_past.groupby("bike")
            .agg(총고장횟수=("bike","count"), 마지막고장일=("date","max"))
            .reset_index().rename(columns={"bike":"자전거번호"})
            .merge(fault_type_mode, on="자전거번호", how="left")
        )
    snap = rental_agg.merge(fault_agg, on="자전거번호", how="left")
    for df in [r7, r30, r90]:
        snap = snap.merge(df, on="자전거번호", how="left")
    for col in ["총고장횟수","최근7일이용거리","최근7일대여횟수","최근30일이용거리","최근30일대여횟수","최근90일이용거리","최근90일대여횟수"]:
        if col in snap.columns:
            snap[col] = snap[col].fillna(0)
    snap["마지막고장후경과일"]   = (snapshot_date - snap["마지막고장일"]).dt.days.fillna(9999)
    snap["고장간격km"]           = ((snap["총이용거리"]/1000) / snap["총고장횟수"].replace(0, np.nan)).fillna(99999)
    snap["고장경험있음"]         = (snap["총고장횟수"] > 0).astype(int)
    snap["마지막대여후경과일"]   = (snapshot_date - snap["마지막대여일"]).dt.days.fillna(9999)
    snap["최근7일_30일대여비율"] = (snap["최근7일대여횟수"] / snap["최근30일대여횟수"].replace(0, np.nan)).fillna(0)
    snap["최근7일_30일거리비율"] = (snap["최근7일이용거리"]  / snap["최근30일이용거리"].replace(0, np.nan)).fillna(0)
    snap["주요고장유형"]         = snap["주요고장유형"].fillna("고장없음")
    future_fault_bikes = set(BR.loc[(BR["date"] > snapshot_date) & (BR["date"] <= future_end), "bike"])
    snap["label"]         = snap["자전거번호"].isin(future_fault_bikes).astype(int)
    snap["snapshot_date"] = snapshot_date
    return snap

@st.cache_resource
def load_model():
    model = joblib.load(base / "xgb_model.pkl")
    with open(base / "features.json", "r", encoding="utf-8") as f:
        features = json.load(f)
    return model, features

model, features = load_model()
rental_daily, BR = load_raw_data()

# ══════════════════════════════════════════════
#  타이틀
# ══════════════════════════════════════════════
st.markdown(
    '<div class="title-wrap"><div class="title-main">자전거 정비 우선순위 예측</div><div class="small-muted" style="margin-top:12px;">고장확률 기반으로 점검 우선순위를 빠르게 확인하세요.</div></div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════
#  사이드바 — 조건 설정
# ══════════════════════════════════════════════
st.sidebar.header("예측 조건 선택")

기준날짜 = st.sidebar.date_input(
    "예측 기준일",
    value=pd.to_datetime("2025-11-30"),
    min_value=pd.to_datetime("2021-01-01"),
    max_value=pd.to_datetime("2025-12-31")
)
top_n    = st.sidebar.slider("표시할 자전거 수", min_value=10, max_value=500, value=100, step=10)
min_prob = st.sidebar.slider("최소 확률 필터", min_value=0, max_value=100, value=0, step=5)

st.sidebar.markdown("""
<div style="background:#F8FAFC;border-radius:14px;padding:16px 18px;margin-top:4px;
            border:1px solid #E2E8F0">
    <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                color:#94A3B8;margin-bottom:12px">위험도 기준</div>
    <div style="display:flex;flex-direction:column;gap:10px">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:8px;height:8px;border-radius:50%;background:#EF4444;flex-shrink:0"></div>
            <span style="font-size:13px;color:#374151;font-weight:500">즉시 점검</span>
            <span style="margin-left:auto;font-size:12px;color:#94A3B8">70%+</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:8px;height:8px;border-radius:50%;background:#F59E0B;flex-shrink:0"></div>
            <span style="font-size:13px;color:#374151;font-weight:500">이번 주 내</span>
            <span style="margin-left:auto;font-size:12px;color:#94A3B8">50~70%</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:8px;height:8px;border-radius:50%;background:#3B82F6;flex-shrink:0"></div>
            <span style="font-size:13px;color:#374151;font-weight:500">이번 달 내</span>
            <span style="margin-left:auto;font-size:12px;color:#94A3B8">30~50%</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:8px;height:8px;border-radius:50%;background:#10B981;flex-shrink:0"></div>
            <span style="font-size:13px;color:#374151;font-weight:500">정기 점검</span>
            <span style="margin-left:auto;font-size:12px;color:#94A3B8">~30%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  예측 실행
# ══════════════════════════════════════════════
with st.spinner("Loading ..."):
    data_filtered = build_snapshot_daily(rental_daily, BR, pd.Timestamp(기준날짜), window_days=30)

if data_filtered.empty:
    st.error("선택한 날짜 기준으로 생성된 데이터가 없습니다.")
    st.stop()

X = data_filtered.reindex(columns=features, fill_value=0)
data_filtered["고장확률"]     = model.predict_proba(X)[:, 1]
data_filtered["고장확률_pct"] = (data_filtered["고장확률"] * 100).round(1)

def 권고사항(p):
    if p >= 70: return "즉시 점검"
    elif p >= 50: return "이번 주 내"
    elif p >= 30: return "이번 달 내"
    return "정기 점검"

def 위험도(p):
    if p >= 70: return "CRITICAL"
    elif p >= 50: return "HIGH"
    elif p >= 30: return "MEDIUM"
    return "LOW"

data_filtered["권고사항"] = data_filtered["고장확률_pct"].apply(권고사항)
data_filtered["위험도"]   = data_filtered["고장확률_pct"].apply(위험도)

rank_df = (
    data_filtered[data_filtered["고장확률_pct"] >= min_prob]
    .sort_values("고장확률_pct", ascending=False)
    .head(top_n)
    .copy()
)
rank_df["순위"] = range(1, len(rank_df) + 1)

# ══════════════════════════════════════════════
#  메인 영역 — 메트릭 + 테이블 + 다운로드
# ══════════════════════════════════════════════

# 섹션 1: 요약 메트릭
st.markdown(
    f"""
    <div class="section-header">
      <div class="section-icon section-icon-blue">📊</div>
      <div>
        <div class="section-title">{기준날짜} 기준 분석 요약</div>
        <div class="section-desc">전체 자전거 대상 고장 확률 분석 결과</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 자전거",    f"{len(data_filtered):,}",                                                                       "분석 대상")
c2.metric("🔴 즉시 점검",  f"{(data_filtered['고장확률_pct'] >= 70).sum():,}",                                               "70% 이상")
c3.metric("🟡 이번 주 내", f"{((data_filtered['고장확률_pct'] >= 50) & (data_filtered['고장확률_pct'] < 70)).sum():,}",     "50~70%")
c4.metric("평균 고장 확률", f"{data_filtered['고장확률_pct'].mean():.1f}%",                                                  "전체 기준")

# 섹션 2: 정비 우선순위 테이블
st.markdown(
    f"""
    <div class="section-header">
      <div class="section-icon section-icon-violet">🔧</div>
      <div>
        <div class="section-title">정비 우선순위 TOP {top_n}</div>
        <div class="section-desc">고장 확률 내림차순 · 기준일 {기준날짜}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# HTML 테이블
BADGE = {
    "CRITICAL": "background:#FEE2E2;color:#991B1B",
    "HIGH":     "background:#FEF3C7;color:#92400E",
    "MEDIUM":   "background:#DBEAFE;color:#1E40AF",
    "LOW":      "background:#D1FAE5;color:#065F46",
}
BAR_COLOR = {
    "CRITICAL": "#EF4444",
    "HIGH":     "#F59E0B",
    "MEDIUM":   "#3B82F6",
    "LOW":      "#10B981",
}

rows = ""
for _, row in rank_df.iterrows():
    prob    = row["고장확률_pct"]
    risk    = row["위험도"]
    color   = BAR_COLOR.get(risk, "#94A3B8")
    badge   = BADGE.get(risk, "")
    elapsed = int(row["마지막고장후경과일"]) if row["마지막고장후경과일"] < 9999 else None
    elapsed_str = f"{elapsed}일 전" if elapsed is not None else "기록없음"
    faults  = int(row["총고장횟수"])

    rows += f"""
    <tr style="border-bottom:1px solid #F1F5F9">
        <td style="padding:12px 16px;color:#94A3B8;font-size:12px;font-weight:600">{int(row['순위'])}</td>
        <td style="padding:12px 16px;font-weight:700;color:#0F1C3F;font-family:monospace;font-size:13px">{row['자전거번호']}</td>
        <td style="padding:12px 16px;min-width:160px">
            <div style="display:flex;align-items:center;gap:10px">
                <div style="flex:1;height:6px;background:#F1F5F9;border-radius:3px;overflow:hidden">
                    <div style="width:{prob:.1f}%;height:100%;background:{color};border-radius:3px"></div>
                </div>
                <span style="font-weight:700;font-size:13px;color:{color};min-width:40px;text-align:right">{prob:.1f}%</span>
            </div>
        </td>
        <td style="padding:12px 16px">
            <span style="display:inline-block;{badge};font-size:11px;font-weight:700;
                         padding:3px 9px;border-radius:6px;letter-spacing:0.04em">{risk}</span>
        </td>
        <td style="padding:12px 16px;color:#475569;font-size:13px">{row['권고사항']}</td>
        <td style="padding:12px 16px;color:#94A3B8;font-size:13px;text-align:center">{elapsed_str}</td>
        <td style="padding:12px 16px;color:#94A3B8;font-size:13px;text-align:center">{faults}회</td>
    </tr>"""

st.markdown(f"""
<div style="background:#FFFFFF;border-radius:16px;overflow:hidden;
            box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.04);margin-bottom:16px">
    <table style="width:100%;border-collapse:collapse;
                  font-family:'Pretendard',-apple-system,sans-serif">
        <thead>
            <tr style="background:#F8FAFC;border-bottom:1.5px solid #E2E8F0">
                <th class="tbl-head">#</th>
                <th class="tbl-head">자전거 번호</th>
                <th class="tbl-head">고장 확률</th>
                <th class="tbl-head">위험도</th>
                <th class="tbl-head">권고사항</th>
                <th class="tbl-head-center">마지막 고장</th>
                <th class="tbl-head-center">누적 고장</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
</div>
""", unsafe_allow_html=True)

# 다운로드 — Excel로 변경 (한글 깨짐 없음)
show_cols = ["순위", "자전거번호", "고장확률_pct", "위험도", "권고사항", "총고장횟수", "마지막고장후경과일"]
final_df  = rank_df[[c for c in show_cols if c in rank_df.columns]].rename(columns={"고장확률_pct": "고장확률(%)"})

buf = io.BytesIO()
final_df.to_excel(buf, index=False, engine="openpyxl")
buf.seek(0)

st.download_button(
    "⬇  정비 목록 Excel 다운로드",
    data=buf,
    file_name=f"따릉이_정비우선순위_{기준날짜}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)