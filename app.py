from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster, HeatMap

try:
    from streamlit_folium import st_folium
except Exception:
    st_folium = None


st.set_page_config(
    page_title="서울시 따릉이 고장 핫스팟 및 예측 대시보드",
    page_icon="🚲",
    layout="wide",
)

st.markdown(
    """
    <style>
      @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
      html, body, [class*="css"], .stApp {font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;}
      .block-container {padding-top: 2.8rem; padding-bottom: 1.2rem;}
      .title-wrap {background: transparent; color: #0f172a; padding: 0; border-radius: 0; margin-bottom: 2px;}
      .title-main {margin:0; font-size:2.0rem; font-weight:800; line-height:1.1; letter-spacing:-0.2px; padding-top:4px;}
      .kpi {background: linear-gradient(135deg,#f8fafc,#e9eef5); padding: 12px; border-radius: 12px; border:1px solid #dbe3eb;}
      .guide-card {background: linear-gradient(135deg,#fffef8,#f4f7ff); border:1px solid #e7edf7; border-radius: 12px; padding: 10px 12px;}
      .legend-wrap {display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 8px 0;}
      .badge {padding:5px 10px; border-radius:999px; color:white; font-weight:600; font-size:0.86rem; display:inline-block;}
      .badge-danger {background:#d62828;}
      .badge-warn {background:#f77f00;}
      .badge-safe {background:#2a9d8f;}
      .small-muted {color:#6b7280; font-size:0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_data"

GRADE_MAP = {"HIGH": "위험", "MID": "주의", "LOW": "보통"}
GRADE_ORDER = ["위험", "주의", "보통"]
GRADE_COLOR = {"위험": "#d62828", "주의": "#f77f00", "보통": "#2a9d8f"}


def pick_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def normalize_station_id(s: pd.Series) -> pd.Series:
    return (
        s.astype("string").str.strip()
        .str.replace(r"\\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .replace({"": pd.NA})
        .str.zfill(5)
    )


def risk_to_ko(series: pd.Series) -> pd.Series:
    return series.astype("string").map(GRADE_MAP).fillna("보통")


@st.cache_data(show_spinner=False)
def read_any(path: Path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_input_files(processed_dir: Path):
    hotspot_path = pick_existing([
        processed_dir / "station_hotspot_map.parquet",
        processed_dir / "station_hotspot_with_coords.parquet",
        processed_dir / "clustered_hotspot.parquet",
        processed_dir / "clustered_hotspot.csv",
    ])
    daily_path = pick_existing([
        processed_dir / "station_hotspot_daily.parquet",
    ])
    repair_path = pick_existing([
        processed_dir / "repair_station_events.parquet",
        processed_dir / "repair_station_events.csv",
    ])
    forecast_station_fallback_path = pick_existing([
        processed_dir / "forecast_station_daily.parquet",
        processed_dir / "forecast_station_daily.csv",
    ])
    forecast_global_fallback_path = pick_existing([
        processed_dir / "forecast_global_daily.parquet",
    ])

    return {
        "hotspot_path": hotspot_path,
        "daily_path": daily_path,
        "repair_path": repair_path,
        "forecast_station_fallback_path": forecast_station_fallback_path,
        "forecast_global_fallback_path": forecast_global_fallback_path,
    }


@st.cache_resource(show_spinner=False)
def load_pickle_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_fault_profile(repair_df: pd.DataFrame) -> pd.DataFrame:
    x = repair_df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    if "추정고장대여소ID" not in x.columns or "고장구분" not in x.columns:
        return pd.DataFrame(columns=["추정고장대여소ID", "대표고장유형", "주요고장TOP3"])

    x["추정고장대여소ID"] = normalize_station_id(x["추정고장대여소ID"])
    x["고장구분"] = x["고장구분"].astype("string").str.strip().replace({"": pd.NA})
    x = x.dropna(subset=["추정고장대여소ID", "고장구분"])
    if x.empty:
        return pd.DataFrame(columns=["추정고장대여소ID", "대표고장유형", "주요고장TOP3"])

    cnt = (
        x.groupby(["추정고장대여소ID", "고장구분"], as_index=False)
        .size()
        .rename(columns={"size": "건수"})
        .sort_values(["추정고장대여소ID", "건수"], ascending=[True, False])
    )
    top1 = (
        cnt.drop_duplicates("추정고장대여소ID")
        [["추정고장대여소ID", "고장구분"]]
        .rename(columns={"고장구분": "대표고장유형"})
    )
    top3 = (
        cnt.groupby("추정고장대여소ID", as_index=False)
        .head(3)
        .groupby("추정고장대여소ID")
        .apply(lambda g: ", ".join((g["고장구분"] + "(" + g["건수"].astype(str) + ")").tolist()))
        .reset_index(name="주요고장TOP3")
    )
    return top1.merge(top3, on="추정고장대여소ID", how="left")


def prepare_hotspot_base(hotspot_raw: pd.DataFrame) -> pd.DataFrame:
    x = hotspot_raw.copy()
    x.columns = [str(c).strip() for c in x.columns]

    need = ["추정고장대여소ID", "위도", "경도", "고장건수"]
    miss = [c for c in need if c not in x.columns]
    if miss:
        raise ValueError(f"핫스팟 입력 파일에 필수 컬럼 누락: {miss}")

    x["추정고장대여소ID"] = normalize_station_id(x["추정고장대여소ID"])
    x["위도"] = pd.to_numeric(x["위도"], errors="coerce")
    x["경도"] = pd.to_numeric(x["경도"], errors="coerce")
    x["고장건수"] = pd.to_numeric(x["고장건수"], errors="coerce")
    x = x.dropna(subset=need)
    x = x[x["고장건수"] > 0].copy()

    # 동일 대여소가 여러 행이면 재집계
    x = (
        x.groupby("추정고장대여소ID", as_index=False)
        .agg(위도=("위도", "mean"), 경도=("경도", "mean"), 고장건수=("고장건수", "sum"))
    )
    x["log_고장건수"] = np.log1p(x["고장건수"]) 
    return x


def apply_cluster_model(hotspot_base: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    x = hotspot_base.copy()

    feature_cols = model_bundle.get("feature_cols", ["위도", "경도", "log_고장건수"])
    scaler = model_bundle.get("scaler")
    dbscan_model = model_bundle.get("dbscan_model")
    kmeans_model = model_bundle.get("kmeans_model")
    default_model = model_bundle.get("ui_default_model", "kmeans")

    X = x[feature_cols].copy()
    X_in = scaler.transform(X) if scaler is not None else X.values

    # DBSCAN: 저장된 파라미터로 재적합
    if dbscan_model is not None and hasattr(dbscan_model, "eps") and hasattr(dbscan_model, "min_samples"):
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=float(dbscan_model.eps), min_samples=int(dbscan_model.min_samples))
        x["dbscan_cluster"] = db.fit_predict(X_in)
    else:
        x["dbscan_cluster"] = -1

    # KMeans: 저장 모델로 predict
    if kmeans_model is not None and hasattr(kmeans_model, "predict"):
        x["kmeans_cluster"] = kmeans_model.predict(X_in)
    else:
        x["kmeans_cluster"] = 0

    if str(default_model).lower().startswith("db"):
        x["ui_cluster"] = x["dbscan_cluster"].astype("string")
    else:
        x["ui_cluster"] = x["kmeans_cluster"].astype("string")

    # 핫스팟 등급
    q90 = x["고장건수"].quantile(0.90)
    q70 = x["고장건수"].quantile(0.70)
    x["핫스팟등급"] = "LOW"
    x.loc[x["고장건수"] >= q70, "핫스팟등급"] = "MID"
    x.loc[x["고장건수"] >= q90, "핫스팟등급"] = "HIGH"

    x["위험등급"] = risk_to_ko(x["핫스팟등급"])
    uniq = sorted(x["ui_cluster"].astype("string").unique().tolist())
    x["권역명"] = x["ui_cluster"].astype("string").map({c: f"권역 {i+1}" for i, c in enumerate(uniq)})
    return x


def prepare_daily(daily_raw: pd.DataFrame) -> pd.DataFrame:
    x = daily_raw.copy()
    x.columns = [str(c).strip() for c in x.columns]
    if "date" not in x.columns and "일자" in x.columns:
        x["date"] = x["일자"]
    need = ["date", "추정고장대여소ID", "고장건수"]
    miss = [c for c in need if c not in x.columns]
    if miss:
        raise ValueError(f"일별 데이터 필수 컬럼 누락: {miss}")

    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["추정고장대여소ID"] = normalize_station_id(x["추정고장대여소ID"])
    x["고장건수"] = pd.to_numeric(x["고장건수"], errors="coerce")
    x = x.dropna(subset=need)
    x = x[x["고장건수"] >= 0].copy()
    return x


def build_global_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    g = (
        daily_df.groupby("date", as_index=False)["고장건수"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
    g = g.set_index("date").reindex(idx).fillna(0.0).rename_axis("date").reset_index()
    return g


def build_feature_row(history_vals: list[float], target_date: pd.Timestamp, start_date: pd.Timestamp):
    arr = np.array(history_vals, dtype=float)

    def get_lag(k):
        return float(arr[-k]) if len(arr) >= k else float(arr.mean())

    return {
        "lag_1": get_lag(1),
        "lag_2": get_lag(2),
        "lag_3": get_lag(3),
        "lag_7": get_lag(7),
        "lag_14": get_lag(14),
        "lag_21": get_lag(21),
        "lag_28": get_lag(28),
        "roll_mean_7": float(arr[-7:].mean()) if len(arr) >= 7 else float(arr.mean()),
        "roll_mean_14": float(arr[-14:].mean()) if len(arr) >= 14 else float(arr.mean()),
        "roll_std_7": float(arr[-7:].std()) if len(arr) >= 7 else float(arr.std()),
        "dow": int(target_date.dayofweek),
        "dom": int(target_date.day),
        "month": int(target_date.month),
        "weekofyear": int(target_date.isocalendar().week),
        "t": int((target_date - start_date).days),
    }


def forecast_with_xgb(
    daily_df: pd.DataFrame,
    station_meta: pd.DataFrame,
    forecast_model_bundle: dict,
    horizon_days: int,
):
    if forecast_model_bundle is None:
        return None, None

    xgb_model = forecast_model_bundle.get("xgb_model")
    feature_cols = forecast_model_bundle.get("xgb_feature_cols")
    if xgb_model is None or not feature_cols:
        return None, None

    global_daily = build_global_daily(daily_df)
    start_date = global_daily["date"].min()
    last_date = global_daily["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    hist = global_daily["고장건수"].astype(float).tolist()
    preds = []
    for d in future_dates:
        feat = build_feature_row(hist, d, start_date)
        x_in = pd.DataFrame([feat])[feature_cols]
        yhat = float(xgb_model.predict(x_in)[0])
        yhat = max(0.0, yhat)
        preds.append(yhat)
        hist.append(yhat)

    forecast_global = pd.DataFrame({"date": future_dates, "yhat_final": preds, "selected_model": "XGBoost"})

    # 대여소 분배 비율 계산 (최근 14일 70% + 최근 56일 30%)
    end_date = daily_df["date"].max()
    short = daily_df[daily_df["date"].between(end_date - pd.Timedelta(days=13), end_date)]
    long = daily_df[daily_df["date"].between(end_date - pd.Timedelta(days=55), end_date)]

    s_short = short.groupby("추정고장대여소ID", as_index=False)["고장건수"].sum().rename(columns={"고장건수": "short_sum"})
    s_long = long.groupby("추정고장대여소ID", as_index=False)["고장건수"].sum().rename(columns={"고장건수": "long_sum"})

    share = station_meta[["추정고장대여소ID"]].drop_duplicates().copy()
    share = share.merge(s_short, on="추정고장대여소ID", how="left")
    share = share.merge(s_long, on="추정고장대여소ID", how="left")
    share[["short_sum", "long_sum"]] = share[["short_sum", "long_sum"]].fillna(0.0)

    if share["short_sum"].sum() > 0:
        share["short_ratio"] = share["short_sum"] / share["short_sum"].sum()
    else:
        share["short_ratio"] = 0.0

    if share["long_sum"].sum() > 0:
        share["long_ratio"] = share["long_sum"] / share["long_sum"].sum()
    else:
        share["long_ratio"] = 0.0

    share["ratio"] = 0.7 * share["short_ratio"] + 0.3 * share["long_ratio"]
    if share["ratio"].sum() == 0:
        share["ratio"] = 1.0 / max(len(share), 1)
    else:
        share["ratio"] = share["ratio"] / share["ratio"].sum()

    fg = forecast_global.copy()
    fg["__k"] = 1
    share2 = share[["추정고장대여소ID", "ratio"]].copy()
    share2["__k"] = 1

    fs = fg.merge(share2, on="__k", how="left").drop(columns=["__k"])
    fs["pred_fault_count"] = fs["yhat_final"] * fs["ratio"]
    fs = fs.merge(station_meta, on="추정고장대여소ID", how="left")

    # 예측 등급
    q90 = fs["pred_fault_count"].quantile(0.90)
    q70 = fs["pred_fault_count"].quantile(0.70)
    fs["pred_risk"] = "LOW"
    fs.loc[fs["pred_fault_count"] >= q70, "pred_risk"] = "MID"
    fs.loc[fs["pred_fault_count"] >= q90, "pred_risk"] = "HIGH"
    fs["예측위험등급"] = risk_to_ko(fs["pred_risk"])

    return fs, forecast_global


def build_map(df, value_col, risk_col, zone_col, color_mode, use_heatmap, use_marker_cluster, title_prefix):
    center = [float(df["위도"].mean()), float(df["경도"].mean())]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron", control_scale=True)

    parent = MarkerCluster(name=f"{title_prefix} 마커") if use_marker_cluster else folium.FeatureGroup(name=f"{title_prefix} 마커")
    parent.add_to(m)

    maxv = max(float(df[value_col].max()), 1.0)
    for _, row in df.iterrows():
        if color_mode == "위험등급":
            color = GRADE_COLOR.get(str(row[risk_col]), "#4b5563")
        else:
            palette = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#8d99ae", "#6a4c93", "#ff7b00", "#00a6fb", "#ef476f", "#118ab2"]
            color = palette[abs(hash(str(row[zone_col]))) % len(palette)]

        radius = 4 + 14 * (float(row[value_col]) / maxv) ** 0.5
        popup_html = f"""
        <b>대여소 ID</b>: {row['추정고장대여소ID']}<br>
        <b>{title_prefix} 고장값</b>: {float(row[value_col]):.2f}<br>
        <b>위험등급</b>: {row[risk_col]}<br>
        <b>권역</b>: {row[zone_col]}<br>
        <b>대표 고장유형</b>: {row.get('대표고장유형', '정보없음')}<br>
        <b>주요 고장 TOP3</b>: {row.get('주요고장TOP3', '정보없음')}
        """

        folium.CircleMarker(
            location=[float(row["위도"]), float(row["경도"])],
            radius=float(radius),
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.72,
            popup=folium.Popup(popup_html, max_width=340),
            tooltip=f"{row['추정고장대여소ID']} | {row.get('대표고장유형','정보없음')}",
        ).add_to(parent)

    if use_heatmap:
        HeatMap(
            df[["위도", "경도", value_col]].values.tolist(),
            name=f"{title_prefix} 히트맵",
            radius=18,
            blur=14,
            min_opacity=0.25,
            max_zoom=13,
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# APP
st.markdown(
    '<div class="title-wrap"><div class="title-main">서울 따릉이 고장 핫스팟 & 시계열 예측 지도</div><div class="small-muted" style="margin-top:6px;">현재 위험 지역과 미래 위험 지역을 한 번에 확인하세요</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="legend-wrap">
      <span class="badge badge-danger">위험</span>
      <span class="badge badge-warn">주의</span>
      <span class="badge badge-safe">보통</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if st_folium is None:
    st.error("`streamlit-folium`이 필요합니다. `pip install streamlit-folium` 후 다시 실행해 주세요.")
    st.stop()

files = load_input_files(PROCESSED_DIR)
if files["hotspot_path"] is None:
    st.error("핫스팟 입력 파일이 없습니다. `station_hotspot_map.parquet` 또는 `clustered_hotspot.parquet`가 필요합니다.")
    st.stop()
if files["daily_path"] is None:
    st.error("시계열 입력 파일이 없습니다. `station_hotspot_daily.parquet`가 필요합니다.")
    st.stop()

# 모델 파일 경로
cluster_model_path = PROCESSED_DIR / "model.pkl"
forecast_model_path = PROCESSED_DIR / "forecast_model.pkl"

if not cluster_model_path.exists():
    st.error(f"필수 모델 파일 누락: `{cluster_model_path}`")
    st.stop()
if not forecast_model_path.exists():
    st.error(f"필수 모델 파일 누락: `{forecast_model_path}`")
    st.stop()

# 데이터 로드
hotspot_raw = read_any(files["hotspot_path"])
daily_raw = read_any(files["daily_path"])
repair_raw = read_any(files["repair_path"]) if files["repair_path"] is not None else None

# 모델 로드
try:
    cluster_model_bundle = load_pickle_model(cluster_model_path)
except Exception as e:
    st.error(f"model.pkl 로드 실패: {e}")
    st.stop()

try:
    forecast_model_bundle = load_pickle_model(forecast_model_path)
except Exception as e:
    st.error(f"forecast_model.pkl 로드 실패: {e}")
    st.info("xgboost 미설치일 수 있습니다. `pip install xgboost` 후 재실행해 주세요.")
    st.stop()

# 전처리 + 모델 적용
hotspot_base = prepare_hotspot_base(hotspot_raw)
hotspot_df = apply_cluster_model(hotspot_base, cluster_model_bundle)

# 고장유형 프로파일 결합
if repair_raw is not None:
    fault_profile = build_fault_profile(repair_raw)
    if not fault_profile.empty:
        hotspot_df = hotspot_df.merge(fault_profile, on="추정고장대여소ID", how="left")
if "대표고장유형" not in hotspot_df.columns:
    hotspot_df["대표고장유형"] = "정보없음"
else:
    hotspot_df["대표고장유형"] = hotspot_df["대표고장유형"].fillna("정보없음")

if "주요고장TOP3" not in hotspot_df.columns:
    hotspot_df["주요고장TOP3"] = "정보없음"
else:
    hotspot_df["주요고장TOP3"] = hotspot_df["주요고장TOP3"].fillna("정보없음")

# 시계열 예측
daily_df = prepare_daily(daily_raw)
station_meta = hotspot_df[["추정고장대여소ID", "위도", "경도", "핫스팟등급", "권역명", "대표고장유형", "주요고장TOP3"]].drop_duplicates()

horizon = st.sidebar.slider("예측 일수", min_value=7, max_value=90, value=30, step=1)
forecast_station_df, forecast_global_df = forecast_with_xgb(
    daily_df=daily_df,
    station_meta=station_meta,
    forecast_model_bundle=forecast_model_bundle,
    horizon_days=horizon,
)

# fallback: 예측 모델 실패 시 사전 저장 파일
if forecast_station_df is None:
    if files["forecast_station_fallback_path"] is not None:
        fs = read_any(files["forecast_station_fallback_path"])
        fs.columns = [str(c).strip() for c in fs.columns]
        fs["date"] = pd.to_datetime(fs["date"], errors="coerce")
        fs["추정고장대여소ID"] = normalize_station_id(fs["추정고장대여소ID"])
        fs["pred_fault_count"] = pd.to_numeric(fs["pred_fault_count"], errors="coerce")
        fs = fs.dropna(subset=["date", "추정고장대여소ID", "pred_fault_count", "위도", "경도"])
        if "예측위험등급" not in fs.columns and "pred_risk" in fs.columns:
            fs["예측위험등급"] = risk_to_ko(fs["pred_risk"])
        if "권역명" not in fs.columns and "권역" in fs.columns:
            fs["권역명"] = "권역 " + fs["권역"].astype("string")
        forecast_station_df = fs
    if files["forecast_global_fallback_path"] is not None:
        fg = read_any(files["forecast_global_fallback_path"])
        fg.columns = [str(c).strip() for c in fg.columns]
        fg["date"] = pd.to_datetime(fg["date"], errors="coerce")
        forecast_global_df = fg

if forecast_station_df is None:
    st.error("예측 데이터 생성에 실패했습니다. forecast_model.pkl 또는 fallback forecast 파일을 확인하세요.")
    st.stop()

# Sidebar - 지도 설정
st.sidebar.header("지도 설정")
color_mode = st.sidebar.selectbox("색상 기준", ["위험등급", "권역"], index=0)
use_heatmap = st.sidebar.toggle("히트맵 표시", value=True)
use_marker_cluster = st.sidebar.toggle("마커 묶기", value=True)
# st.sidebar.caption("`위험등급`: 위험/주의/보통 색상, `권역`: 군집별 색상")

# KPI
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
next7_total = forecast_station_df[forecast_station_df["date"] <= (forecast_station_df["date"].min() + pd.Timedelta(days=6))]["pred_fault_count"].sum()
k1.markdown(f'<div class="kpi"><b>현재 대여소 수</b><br><span style="font-size:1.4rem;">{len(hotspot_df):,}</span></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi"><b>현재 총 고장건수</b><br><span style="font-size:1.4rem;">{int(hotspot_df["고장건수"].sum()):,}</span></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi"><b>예측 {horizon}일 총 고장건수</b><br><span style="font-size:1.4rem;">{int(round(forecast_station_df["pred_fault_count"].sum())):,}</span></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi"><b>예측 7일 고장 합</b><br><span style="font-size:1.4rem;">{int(round(next7_total)):,}</span></div>', unsafe_allow_html=True)

# Tabs
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
tab_now, tab_pred = st.tabs(["현재 핫스팟 보기", "미래 예측 보기"])

with tab_now:
    st.subheader("현재 고장 핫스팟 지도")
    st.caption("지도 원의 크기는 고장건수, 색상은 선택한 기준(위험등급/권역)을 의미합니다.")
    c1, c2, c3 = st.columns([1.0, 1.35, 1.0])
    risk_sel = c1.multiselect("위험등급 필터", GRADE_ORDER, default=GRADE_ORDER)
    zones = sorted(hotspot_df["권역명"].unique().tolist())
    zone_label_map = {z: z.replace("권역 ", "권역") for z in zones}  # 권역 1 -> 권역1
    inv_zone_label_map = {v: k for k, v in zone_label_map.items()}
    zone_labels = [zone_label_map[z] for z in zones]
    zone_sel_labels = c2.multiselect("권역 필터", zone_labels, default=zone_labels)
    zone_sel = [inv_zone_label_map[z] for z in zone_sel_labels]
    fault_types = sorted([x for x in hotspot_df["대표고장유형"].astype(str).unique().tolist() if x != "정보없음"])
    fault_sel = c3.selectbox("대표 고장유형", ["전체"] + fault_types if fault_types else ["전체"], index=0)

    minv, maxv = int(hotspot_df["고장건수"].min()), int(hotspot_df["고장건수"].max())
    vr = st.slider("고장건수 범위", minv, maxv, (minv, maxv), key="now_range")

    now_f = hotspot_df[
        hotspot_df["위험등급"].isin(risk_sel)
        & hotspot_df["권역명"].isin(zone_sel)
        & hotspot_df["고장건수"].between(vr[0], vr[1])
    ].copy()
    if fault_sel != "전체":
        now_f = now_f[now_f["대표고장유형"] == fault_sel]

    if now_f.empty:
        st.warning("현재 탭: 조건에 맞는 데이터가 없습니다.")
    else:
        map_now = build_map(now_f, "고장건수", "위험등급", "권역명", color_mode, use_heatmap, use_marker_cluster, "현재")
        st_folium(map_now, width=None, height=640, returned_objects=[])

        l, r = st.columns([1.35, 1])
        with l:
            st.markdown("**상위 핫스팟 Top 20**")
            cols = ["추정고장대여소ID", "고장건수", "대표고장유형", "주요고장TOP3", "위험등급", "권역명"]
            st.dataframe(now_f.sort_values("고장건수", ascending=False).head(20)[cols], use_container_width=True)
        with r:
            st.markdown("**대표 고장유형 분포 (Top 10)**")
            fb = now_f.groupby("대표고장유형", as_index=False)["고장건수"].sum().sort_values("고장건수", ascending=False).head(10)
            st.bar_chart(fb.set_index("대표고장유형"))

with tab_pred:
    st.subheader("미래 예측 핫스팟 지도")
    st.caption("선택한 날짜에 고장이 집중될 가능성이 높은 대여소를 표시합니다.")

    dates = sorted(pd.to_datetime(forecast_station_df["date"]).dropna().dt.date.unique().tolist())
    if not dates:
        st.warning("예측 날짜가 없습니다.")
    else:
        c1, c2, c3 = st.columns([0.95, 1.0, 1.45])
        dsel = c1.selectbox("예측 날짜", dates, index=0)
        prisk = c2.multiselect("예측 위험등급 필터", GRADE_ORDER, default=GRADE_ORDER)
        pzones = sorted(forecast_station_df["권역명"].astype(str).unique().tolist())
        pzone_label_map = {z: z.replace("권역 ", "권역") for z in pzones}  # 권역 1 -> 권역1
        inv_pzone_label_map = {v: k for k, v in pzone_label_map.items()}
        pzone_labels = [pzone_label_map[z] for z in pzones]
        pzone_sel_labels = c3.multiselect("예측 권역 필터", pzone_labels, default=pzone_labels)
        pzone_sel = [inv_pzone_label_map[z] for z in pzone_sel_labels]

        day = forecast_station_df[pd.to_datetime(forecast_station_df["date"]).dt.date == dsel].copy()
        pmin, pmax = float(day["pred_fault_count"].min()), float(day["pred_fault_count"].max())
        p_range = st.slider("예측 고장값 범위", pmin, pmax, (pmin, pmax), key="pred_range")

        pred_f = day[
            day["예측위험등급"].isin(prisk)
            & day["권역명"].isin(pzone_sel)
            & day["pred_fault_count"].between(p_range[0], p_range[1])
        ].copy()

        if pred_f.empty:
            st.warning("예측 탭: 조건에 맞는 데이터가 없습니다.")
        else:
            map_pred = build_map(pred_f, "pred_fault_count", "예측위험등급", "권역명", color_mode, use_heatmap, use_marker_cluster, "예측")
            st_folium(map_pred, width=None, height=640, returned_objects=[])

            l2, r2 = st.columns([1.35, 1])
            with l2:
                st.markdown("**예측 상위 대여소 Top 20**")
                cols2 = ["추정고장대여소ID", "pred_fault_count", "대표고장유형", "주요고장TOP3", "예측위험등급", "권역명"]
                st.dataframe(pred_f.sort_values("pred_fault_count", ascending=False).head(20)[cols2], use_container_width=True)
            with r2:
                st.markdown("**예측 대표 고장유형 분포 (Top 10)**")
                pfb = pred_f.groupby("대표고장유형", as_index=False)["pred_fault_count"].sum().sort_values("pred_fault_count", ascending=False).head(10)
                st.bar_chart(pfb.set_index("대표고장유형"))

            if forecast_global_df is not None and "date" in forecast_global_df.columns and "yhat_final" in forecast_global_df.columns:
                fg = forecast_global_df.copy()
                fg["date"] = pd.to_datetime(fg["date"], errors="coerce")
                fg["yhat_final"] = pd.to_numeric(fg["yhat_final"], errors="coerce")
                fg = fg.dropna(subset=["date", "yhat_final"]).sort_values("date")
                st.markdown("**미래 전체 고장건수 추이**")
                st.line_chart(fg.set_index("date")["yhat_final"])
