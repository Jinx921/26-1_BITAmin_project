import streamlit as st

st.set_page_config(
    page_title="서울시 따릉이 고장 분석 통합 포털",
    page_icon="🚲",
    layout="wide",
)

st.markdown(
    """
    <style>
      @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
      html, body, [class*="css"], .stApp {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }
      .block-container {
        padding-top: 3.5rem;
        padding-bottom: 1.2rem;
        max-width: 1080px;
      }
      header[data-testid="stHeader"] {
        background: transparent;
      }

      /* ── Hero ── */
      .hero {
        position: relative;
        background: linear-gradient(135deg, #eef4ff 0%, #f0f7ff 40%, #faf5ff 100%);
        border: 1px solid #dbe4f0;
        border-radius: 20px;
        padding: 32px 32px 28px 32px;
        overflow: hidden;
      }
      .hero::before {
        content: "";
        position: absolute;
        right: -40px;
        top: -40px;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%);
        pointer-events: none;
      }
      .hero::after {
        content: "";
        position: absolute;
        left: -30px;
        bottom: -30px;
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59,130,246,0.07) 0%, transparent 70%);
        pointer-events: none;
      }
      .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: white;
        border: 1px solid #e0e7ff;
        border-radius: 999px;
        padding: 5px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #4f46e5;
        letter-spacing: 0.3px;
        margin-bottom: 14px;
      }
      .hero-title {
        position: relative;
        margin: 0 !important;
        font-size: 2.30rem !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        letter-spacing: -0.5px;
        color: #0f172a !important;
        padding: 7px 0 !important;
      }
      .hero-sub {
        position: relative;
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 8px;
        line-height: 1.5;
      }
      .hero-line {
        margin-top: 40px;
        width: 50px;
        height: 3px;
        border-radius: 999px;
        background: linear-gradient(90deg, #6366f1 0%, #3b82f6 50%, #06b6d4 100%);
      }

      /* ── Section Label ── */
      .section-label {
        font-size: 0.90rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 10px;
        margin-bottom: 14px;
      }

      /* ── Cards ── */
      .card {
        position: relative;
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 24px 22px 20px 22px;
        transition: all .25s cubic-bezier(.4,0,.2,1);
        min-height: 150px;
      }
      .card:hover {
        transform: translateY(-3px);
        border-color: #c7d2fe;
        box-shadow: 0 12px 28px rgba(99, 102, 241, 0.10), 0 4px 10px rgba(0,0,0,0.04);
      }
      .card-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        margin-bottom: 14px;
      }
      .card-icon-blue { background: #eff6ff; }
      .card-icon-violet { background: #f5f3ff; }
      .card-icon-cyan { background: #ecfeff; }
      .card-title {
        color: #111827;
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.25;
        margin: 0 0 6px 0;
      }
      .card-sub {
        color: #9ca3af;
        font-size: 0.85rem;
        margin: 0;
        line-height: 1.45;
      }
      .card-tag {
        display: inline-block;
        margin-top: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #6366f1;
        background: #eef2ff;
        border-radius: 6px;
        padding: 3px 8px;
      }

      /* ── Buttons ── */
      .stButton > button {
        border-radius: 12px;
        border: 1px solid #e2e5ea;
        height: 44px;
        font-weight: 600;
        font-size: 0.92rem;
        background: linear-gradient(135deg, #fafbff 0%, #f8f9fc 100%);
        color: #374151;
        transition: all .2s ease;
      }
      .stButton > button:hover {
        border-color: #a5b4fc;
        color: #4338ca;
        background: linear-gradient(135deg, #eef2ff 0%, #f0f0ff 100%);
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.12);
      }

      /* ── Footer ── */
      .footer {
        text-align: center;
        color: #cbd5e1;
        font-size: 0.78rem;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #f1f5f9;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero ──
st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">따릉이 고장 분석</div>
      <h1 class="hero-title">서울시 따릉이 고장 분석 통합 포털</h1>
      <div class="hero-sub">고장 핫스팟 시각화 · 정비 우선순위 예측 · 날씨 기반 고장 분석</div>
      <div class="hero-line"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Dashboard</div>", unsafe_allow_html=True)

# ── Cards ──
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown(
        """
        <div class="card">
          <div class="card-icon card-icon-blue">🗺️</div>
          <div class="card-title">핫스팟 + 예측 지도</div>
          <div class="card-sub">대여소별 고장 밀집 지역을 지도 위에 시각화하고, 미래 고장 발생을 예측합니다.</div>
          <div class="card-tag">공간 분석</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    if st.button("열기", use_container_width=True, key="go_me"):
        st.switch_page("pages/hotspot_app.py")

with c2:
    st.markdown(
        """
        <div class="card">
          <div class="card-icon card-icon-violet">🔧</div>
          <div class="card-title">정비 우선순위 예측</div>
          <div class="card-sub">자전거 단위의 고장 확률을 산출하고, 선제적 점검 대상을 권고합니다.</div>
          <div class="card-tag">예측 모델</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    if st.button("열기", use_container_width=True, key="go_hs"):
        st.switch_page("pages/repair_app.py")

with c3:
    st.markdown(
        """
        <div class="card">
          <div class="card-icon card-icon-cyan">🌦️</div>
          <div class="card-title">날씨 기반 고장 예측</div>
          <div class="card-sub">기상 조건과 공휴일 정보를 활용하여 유형별 고장 건수를 예측합니다.</div>
          <div class="card-tag">기상 연동</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    if st.button("열기", use_container_width=True, key="go_p3"):
        st.switch_page("pages/weather_app.py")

# ── Footer ──
# st.markdown(
   # "<div class='footer'>BITAmin TS Project · 2026-1</div>",
   # unsafe_allow_html=True,
# )