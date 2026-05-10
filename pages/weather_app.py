import streamlit as st
import pandas as pd
import numpy as np
import pickle
import holidays
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="따릉이 고장 예측 시스템",
    page_icon="🚲",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YEWON_DIR = PROJECT_ROOT / "processed_data" / "yewon"

st.markdown(
    """
    <style>
      @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
      html, body, [class*="css"], .stApp {
          font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }
      .block-container {
          padding-top: 2.8rem;
          padding-bottom: 1.2rem;
          max-width: 1280px;
      }
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
      .small-muted { color:#6b7280; font-size:1.00rem; }
      .legend-wrap { display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 8px 0; }
      .badge {
          padding:5px 10px;
          border-radius:999px;
          color:white;
          font-weight:700;
          font-size:0.86rem;
          display:inline-block;
      }
      .badge-danger { background:#d62828; }
      .badge-warn { background:#f77f00; }
      .badge-safe { background:#2a9d8f; }
      [data-testid="metric-container"] {
          background: linear-gradient(135deg,#f8fafc,#e9eef5) !important;
          border: 1px solid #dbe3eb !important;
          border-radius: 12px !important;
          box-shadow: none !important;
      }
      [data-testid="stMetricLabel"] { color:#6b7280 !important; font-size:11px !important; font-weight:700 !important; }
      [data-testid="stMetricValue"] { color:#0f172a !important; font-weight:800 !important; }
      .stPlotlyChart {
          border:1px solid #e5e7eb;
          border-radius:12px;
          padding:8px;
          background:white;
      }

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
      .section-icon-green { background: #f0fdf4; }
      .section-icon-orange { background: #fff7ed; }
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

      /* ── Result Table ── */
      .result-table-wrap {
          background: white;
          border-radius: 14px;
          overflow: hidden;
          border: 1px solid #e5e7eb;
          margin-bottom: 8px;
      }
      .result-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
      }
      .result-table thead tr {
          background: #f8fafc;
          border-bottom: 1.5px solid #e2e8f0;
      }
      .result-table th {
          padding: 13px 16px;
          text-align: left;
          font-size: 12.5px;
          font-weight: 700;
          letter-spacing: 0.04em;
          color: #475569;
      }
      .result-table td {
          padding: 12px 16px;
          color: #334155;
          border-bottom: 1px solid #f1f5f9;
      }
      .result-table tbody tr:last-child td {
          border-bottom: none;
      }
      .result-table tbody tr:hover {
          background: #f8fafc;
      }
      .risk-badge {
          display: inline-block;
          padding: 3px 10px;
          border-radius: 6px;
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.03em;
      }
      .risk-high { background: #fee2e2; color: #991b1b; }
      .risk-mid  { background: #fef3c7; color: #92400e; }
      .risk-low  { background: #d1fae5; color: #065f46; }

      /* ── Priority Cards ── */
      .priority-card {
          background: white;
          border: 1px solid #e5e7eb;
          border-radius: 14px;
          padding: 18px 20px;
          margin-bottom: 10px;
          transition: all .2s ease;
      }
      .priority-card:hover {
          border-color: #cbd5e1;
          box-shadow: 0 4px 12px rgba(0,0,0,0.04);
      }
      .priority-rank {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 26px;
          height: 26px;
          border-radius: 8px;
          font-size: 12px;
          font-weight: 800;
          margin-right: 10px;
          flex-shrink: 0;
      }
      .rank-1 { background: #fee2e2; color: #dc2626; }
      .rank-2 { background: #fef3c7; color: #d97706; }
      .rank-3 { background: #dbeafe; color: #2563eb; }
      .rank-other { background: #f1f5f9; color: #64748b; }
      .priority-header {
          display: flex;
          align-items: center;
          margin-bottom: 8px;
      }
      .priority-type {
          font-size: 0.95rem;
          font-weight: 700;
          color: #0f172a;
      }
      .priority-count {
          margin-left: auto;
          font-size: 1.15rem;
          font-weight: 800;
          color: #0f172a;
      }
      .priority-count span {
          font-size: 0.8rem;
          font-weight: 500;
          color: #94a3b8;
      }
      .priority-advice {
          font-size: 0.83rem;
          color: #64748b;
          line-height: 1.55;
          margin-top: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title-wrap"><div class="title-main">날씨 기반 고장 예측</div><div class="small-muted" style="margin-top:12px;">기상·공휴일·연휴 특성을 반영해 고장 유형별 예상 건수를 확인하세요.</div></div>',
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    with open(YEWON_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_weather_all():
    weather_path = YEWON_DIR / "2026기상청날씨데이터.csv"
    weather_all = pd.read_csv(weather_path, encoding="cp949")
    weather_all["일시"] = pd.to_datetime(weather_all["일시"], errors="coerce")

    weather_all = weather_all.rename(columns={
        "일시": "ds",
        "평균기온(°C)": "avg_temp",
        "평균기온(℃)": "avg_temp",
        "일강수량(mm)": "rainfall",
        "일 최심적설(cm)": "snow"
    })

    weather_all = weather_all[[
        "ds", "avg_temp", "rainfall", "snow"
    ]].copy()

    return weather_all

def prepare_weather_2026(weather_all):
    weather_2026 = weather_all[
        (weather_all["ds"] >= "2026-01-01") &
        (weather_all["ds"] <= "2026-12-31")
    ].copy()

    weather_2026 = weather_2026.sort_values("ds").reset_index(drop=True)

    weather_2026["rainfall"] = weather_2026["rainfall"].fillna(0)
    weather_2026["snow"] = weather_2026["snow"].fillna(0)
    weather_2026["avg_temp"] = weather_2026["avg_temp"].interpolate()

    kr_holidays_2026 = holidays.KR(years=[2026])

    weather_2026["is_holiday"] = weather_2026["ds"].apply(
        lambda x: 1 if x.date() in kr_holidays_2026 else 0
    )

    weather_2026["is_weekend"] = weather_2026["ds"].dt.weekday.apply(
        lambda x: 1 if x >= 5 else 0
    )

    weather_2026["weekday"] = weather_2026["ds"].dt.weekday
    weather_2026["month"] = weather_2026["ds"].dt.month

    weather_2026["is_rest_day"] = (
        (weather_2026["is_holiday"] == 1) |
        (weather_2026["is_weekend"] == 1)
    ).astype(int)

    weather_2026["rest_group"] = (
        weather_2026["is_rest_day"].ne(weather_2026["is_rest_day"].shift())
    ).cumsum()

    rest_lengths = weather_2026.groupby("rest_group")["is_rest_day"].transform("sum")

    weather_2026["is_long_holiday"] = np.where(
        (weather_2026["is_rest_day"] == 1) & (rest_lengths >= 2),
        1,
        0
    )

    weather_2026 = weather_2026.drop(columns=["rest_group"])

    weather_2026["avg_temp_7d"] = weather_2026["avg_temp"].rolling(7, min_periods=1).mean()
    weather_2026["avg_temp_30d"] = weather_2026["avg_temp"].rolling(30, min_periods=1).mean()
    weather_2026["rainfall_7d"] = weather_2026["rainfall"].rolling(7, min_periods=1).sum()
    weather_2026["rainfall_30d"] = weather_2026["rainfall"].rolling(30, min_periods=1).sum()
    weather_2026["snow_7d"] = weather_2026["snow"].rolling(7, min_periods=1).sum()

    return weather_2026

def get_risk_level(repair_type, yhat, risk_thresholds):
    high = risk_thresholds[repair_type]["high"]
    medium = risk_thresholds[repair_type]["medium"]

    if yhat >= high:
        return "높음"
    elif yhat >= medium:
        return "보통"
    else:
        return "낮음"

def make_advice(row, risk_thresholds):
    repair_type = row["repair_type"]
    yhat = row["yhat"]
    avg_temp = row["avg_temp"]
    rainfall_7d = row["rainfall_7d"]
    is_holiday = row["is_holiday"]
    is_weekend = row["is_weekend"]
    is_long_holiday = row["is_long_holiday"]

    advice = []

    risk = get_risk_level(repair_type, yhat, risk_thresholds)

    if repair_type == "전체":
        advice.append("전체 고장 예측값은 부품별 예측 결과를 합산한 값입니다.")
        advice.append("예상 건수가 높으면 고장 유형별 우선순위를 확인하세요.")

    elif repair_type == "타이어":
        if avg_temp >= 25:
            advice.append("고온으로 인한 타이어 팽창·마모 위험이 큽니다.")
        if rainfall_7d >= 30:
            advice.append("최근 강수 누적으로 노면 이물질에 의한 타이어 손상 가능성이 있습니다.")
        advice.append("타이어 공기압과 마모 상태를 우선 점검하세요.")

    elif repair_type == "체인":
        if rainfall_7d >= 20:
            advice.append("최근 비로 인해 체인 윤활 저하와 부식 가능성이 있습니다.")
        if avg_temp >= 25:
            advice.append("이용량 증가로 체인 마모 신고가 늘어날 수 있습니다.")
        advice.append("체인 장력과 윤활 상태를 점검하세요.")

    elif repair_type == "페달":
        if avg_temp >= 25:
            advice.append("고온과 이용량 증가로 페달 베어링 마모 가능성이 있습니다.")
        if rainfall_7d >= 30:
            advice.append("강수 이후 수분과 이물질 유입으로 페달 고장이 지연 발생할 수 있습니다.")
        advice.append("페달 유격과 회전 상태를 점검하세요.")

    elif repair_type == "단말기":
        if avg_temp >= 28:
            advice.append("고온으로 인한 전자장치 오류 가능성이 있습니다.")
        if rainfall_7d >= 20:
            advice.append("습기 유입으로 단말기 화면·결제·잠금 오류 가능성이 있습니다.")
        advice.append("단말기 화면, 배터리, 잠금장치를 점검하세요.")

    elif repair_type == "안장":
        advice.append("안장 고장은 날씨보다 이용량 변화의 영향을 더 크게 받을 수 있습니다.")
        advice.append("안장 흔들림, 높이 조절부, 파손 여부를 점검하세요.")

    else:
        advice.append("해당 고장 유형의 주요 부품을 점검하세요.")

    if is_long_holiday == 1:
        advice.append("연휴 기간에는 당일 신고는 줄 수 있으나, 연휴 이후 누적 고장이 나타날 수 있습니다.")
    elif is_weekend == 1:
        advice.append("주말에는 신고가 줄지만, 2~4일 뒤 평일에 고장 신고가 증가할 수 있습니다.")
    elif is_holiday == 1:
        advice.append("공휴일 당일 신고는 낮지만, 2~3일 뒤 회복성 신고가 발생할 수 있습니다.")

    return risk, " ".join(advice)

model_bundle = load_model()
weather_all = load_weather_all()
weather_2026 = prepare_weather_2026(weather_all)

if len(weather_2026) == 0:
    st.error("2026년 날씨 데이터가 없습니다. 2026기상청날씨데이터.csv에 2026년 데이터가 포함되어 있는지 확인하세요.")
    st.stop()

models = model_bundle["models"]
feature_cols = model_bundle["feature_cols"]
repair_types = model_bundle["repair_types"]
display_repair_types = model_bundle.get("display_repair_types", ["전체"] + repair_types)
risk_thresholds = model_bundle["risk_thresholds"]

st.sidebar.header("예측 조건 선택")

selected_date = st.sidebar.date_input(
    "예측 날짜",
    value=weather_2026["ds"].min().date(),
    min_value=weather_2026["ds"].min().date(),
    max_value=weather_2026["ds"].max().date()
)

selected_types = st.sidebar.multiselect(
    "고장 유형 선택",
    options=display_repair_types,
    default=display_repair_types
)

prediction_results = []

for repair_type in repair_types:
    model = models[repair_type]
    future = weather_2026[["ds"] + feature_cols].copy()
    forecast = model.predict(future)

    temp_result = pd.DataFrame({
        "ds": weather_2026["ds"],
        "repair_type": repair_type,
        "yhat": forecast["yhat"],
        "yhat_lower": forecast["yhat_lower"],
        "yhat_upper": forecast["yhat_upper"]
    })

    prediction_results.append(temp_result)

pred_2026_parts = pd.concat(prediction_results, ignore_index=True)

pred_2026_parts["yhat"] = pred_2026_parts["yhat"].clip(lower=0)
pred_2026_parts["yhat_lower"] = pred_2026_parts["yhat_lower"].clip(lower=0)
pred_2026_parts["yhat_upper"] = pred_2026_parts["yhat_upper"].clip(lower=0)

pred_2026_total = (
    pred_2026_parts
    .groupby("ds")
    .agg(
        yhat=("yhat", "sum"),
        yhat_lower=("yhat_lower", "sum"),
        yhat_upper=("yhat_upper", "sum")
    )
    .reset_index()
)

pred_2026_total["repair_type"] = "전체"

pred_2026 = pd.concat(
    [pred_2026_total, pred_2026_parts],
    ignore_index=True
)

pred_with_weather = pd.merge(
    pred_2026,
    weather_2026,
    on="ds",
    how="left"
)

advice_results = pred_with_weather.apply(
    lambda row: make_advice(row, risk_thresholds),
    axis=1
)

pred_with_weather["risk_level"] = [x[0] for x in advice_results]
pred_with_weather["advice"] = [x[1] for x in advice_results]

selected_date_pd = pd.to_datetime(selected_date)

today_pred = pred_with_weather[
    (pred_with_weather["ds"] == selected_date_pd) &
    (pred_with_weather["repair_type"].isin(selected_types))
].copy()

st.markdown(
    f"""
    <div class="section-header">
      <div class="section-icon section-icon-blue">📅</div>
      <div>
        <div class="section-title">{selected_date} 예측 결과</div>
        <div class="section-desc">선택 날짜의 기상 조건과 고장 유형별 예측값</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

weather_today = weather_2026[weather_2026["ds"] == selected_date_pd].iloc[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("평균기온", f"{weather_today['avg_temp']:.1f}℃")
col2.metric("강수량", f"{weather_today['rainfall']:.1f}mm")
col3.metric("최근 7일 강수량", f"{weather_today['rainfall_7d']:.1f}mm")
col4.metric("적설량", f"{weather_today['snow']:.1f}cm")

show_df = today_pred[[
    "repair_type", "yhat", "yhat_lower", "yhat_upper", "risk_level", "advice"
]].copy()

show_df = show_df.rename(columns={
    "repair_type": "고장 유형",
    "yhat": "예상 고장 건수",
    "yhat_lower": "예측 하한",
    "yhat_upper": "예측 상한",
    "risk_level": "위험도",
    "advice": "정비 조언"
})

show_df["예상 고장 건수"] = show_df["예상 고장 건수"].round(0).astype(int)
show_df["예측 하한"] = show_df["예측 하한"].round(0).astype(int)
show_df["예측 상한"] = show_df["예측 상한"].round(0).astype(int)

RISK_CLASS = {"높음": "risk-high", "보통": "risk-mid", "낮음": "risk-low"}

table_rows = ""
for _, r in show_df.iterrows():
    rc = RISK_CLASS.get(r["위험도"], "risk-low")
    table_rows += f"""
    <tr>
      <td style="font-weight:700">{r["고장 유형"]}</td>
      <td style="font-weight:700;color:#0f172a">{r["예상 고장 건수"]}</td>
      <td>{r["예측 하한"]} ~ {r["예측 상한"]}</td>
      <td><span class="risk-badge {rc}">{r["위험도"]}</span></td>
      <td style="font-size:12px;color:#64748b;max-width:360px">{r["정비 조언"]}</td>
    </tr>"""

st.markdown(
    f"""
    <div class="result-table-wrap">
      <table class="result-table">
        <thead>
          <tr>
            <th>고장 유형</th>
            <th>예상 건수</th>
            <th>예측 범위</th>
            <th>위험도</th>
            <th>정비 조언</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-header">
      <div class="section-icon section-icon-green">📈</div>
      <div>
        <div class="section-title">2026년 고장 유형별 예측 추이</div>
        <div class="section-desc">연간 시계열 기반 유형별 예상 고장 건수 변화</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

chart_df = pred_with_weather[
    pred_with_weather["repair_type"].isin(selected_types)
].copy()

fig = px.line(
    chart_df,
    x="ds",
    y="yhat",
    color="repair_type",
    labels={
        "ds": "날짜",
        "yhat": "예상 고장 건수",
        "repair_type": "고장 유형"
    }
)
fig.update_layout(
    plot_bgcolor="white",
    margin=dict(t=40, b=40, r=30),
    legend=dict(
        orientation="h",
        yanchor="top", y=1.12,
        xanchor="right", x=1,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e5e7eb",
        borderwidth=1,
    ),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <div class="section-header">
      <div class="section-icon section-icon-orange">🔧</div>
      <div>
        <div class="section-title">정비 우선순위</div>
        <div class="section-desc">예상 고장 건수 기준 내림차순 정렬</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

priority_df = today_pred.sort_values("yhat", ascending=False).copy()
priority_df["rank"] = range(1, len(priority_df) + 1)

for _, row in priority_df.iterrows():
    rank = int(row['rank'])
    if rank == 1:
        rank_cls = "rank-1"
    elif rank == 2:
        rank_cls = "rank-2"
    elif rank == 3:
        rank_cls = "rank-3"
    else:
        rank_cls = "rank-other"

    rc = RISK_CLASS.get(row['risk_level'], "risk-low")

    st.markdown(
        f"""
        <div class="priority-card">
          <div class="priority-header">
            <span class="priority-rank {rank_cls}">{rank}</span>
            <span class="priority-type">{row['repair_type']}</span>
            <span class="risk-badge {rc}" style="margin-left:10px">{row['risk_level']}</span>
            <div class="priority-count">{row['yhat']:.0f}<span> 건</span></div>
          </div>
          <div class="priority-advice">{row['advice']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )