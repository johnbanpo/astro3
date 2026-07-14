import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 페이지 설정 ---
st.set_page_config(page_title="지구 자기장 다이나모 시뮬레이터 Pro", layout="wide")

st.title("🌋 다이나모 이론: 지구 자기장 생성 및 메커니즘 시뮬레이터")
st.markdown("지구 외핵의 물리적 현상을 수치적으로 구현하고, 각 효과의 개별 메커니즘을 시각화하는 인터랙티브 모델입니다.")

# --- 탭 구성 ---
tab1, tab2 = st.tabs(["📊 통합 다이나모 시뮬레이션", "🌪️ 메커니즘 개별 시각화 (α & Ω)"])

# ==========================================
# TAB 1: 통합 다이나모 시뮬레이션 (자기력선 추가)
# ==========================================
with tab1:
    st.subheader("🌐 주 자기장(Bp)과 내부 자기장(Bt)의 상호작용 및 자기력선")
    
    # 사이드바 제어 요소
    st.sidebar.header("⚙️ 1. 통합 시뮬레이션 계수 제어")
    alpha = st.sidebar.slider("알파 효과 강도 (α-Effect)", 0.0, 5.0, 2.5, 0.1)
    omega = st.sidebar.slider("오메가 효과 강도 (Ω-Effect)", 0.0, 10.0, 6.0, 0.1)
    eta = st.sidebar.slider("자기 확산도 (η)", 0.1, 2.0, 1.2, 0.05)
    grid_size = st.sidebar.slider("그리드 해상도 (Grid)", 20, 50, 40, 5)

    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        start_btn = st.button("▶️ 시작", key="start_t1")
    with col2:
        stop_btn = st.button("⏸️ 일시정지", key="stop_t1")
    with col3:
        reset_btn = st.button("🔄 초기화", key="reset_t1")

    # 세션 상태 초기화
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "running" not in st.session_state:
        st.session_state.running = False

    if start_btn: st.session_state.running = True
    if stop_btn: st.session_state.running = False
    if reset_btn:
        st.session_state.step = 0
        st.session_state.running = False

    # 그리드 및 초기 물리장 설정
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # 지구 핵 내부 영역 마스크 (R <= 1)
    core_mask = R <= 1.0

    # 초기 폴로이달(Bp), 토로이달(Bt) 장 정의 (초기 쌍극자 형태 모사)
    Bp = np.exp(-R**2) * Y * core_mask
    Bt = np.zeros_like(Bp)

    # 시간 발전 (시뮬레이션 루프)
    dt = 0.01
    
    # 실제 러닝 스텝 반복
    if st.session_state.running:
        # 간단한 의사 시간 발전 (유한차분법의 단순화 모델)
        for _ in range(st.session_state.step % 50 + 1):
            # 오메가 효과: Bp가 차동 회전(Ω)에 의해 Bt를 생성
            dBt = omega * Bp * (1 - R**2) * core_mask - eta * Bt
            Bt += dBt * dt
            
            # 알파 효과: Bt가 회오리 대류(α)에 의해 Bp를 생성
            dBp = alpha * Bt * core_mask - eta * Bp
            Bp += dBp * dt
            
        st.session_state.step += 1

    # 시각화 그래프 작성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 폴로이달 성분 + 자기력선 (Streamplot)
    im1 = ax1.imshow(Bp, extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
    # 자기력선 유도 (Bp의 경도 성분을 이용해 가상의 벡터장 생성)
    dY, dX = np.gradient(Bp)
    # 외곽 및 중심부에 겉보기 벡터 흐름(자기력선) 표현
    ax1.streamplot(X, Y, -dY, dX, color='black', linewidth=0.8, density=1.0, arrowstyle='->')
    # 지구 내핵/외핵 경계선 표시
    circle1 = plt.Circle((0, 0), 1.0, color='yellow', fill=False, linestyle='--', linewidth=1.5, label='Outer Core')
    ax1.add_patch(circle1)
    ax1.set_title(f"🌐 주 자기장: 폴로이달 성분 ($B_p$) & 자기력선 (Step: {st.session_state.step})", fontsize=12)
    fig.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.8)

    # 2. 토로이달 성분 (내부 유체 감김 현상) + 등고선 (Contour)
    im2 = ax2.imshow(Bt, extent=[-2, 2, -2, 2], origin='lower', cmap='twilight', vmin=-2, vmax=2)
    ax2.contour(X, Y, Bt, levels=8, colors='white', alpha=0.5, linewidths=0.7)
    circle2 = plt.Circle((0, 0), 1.0, color='yellow', fill=False, linestyle='--', linewidth=1.5)
    ax2.add_patch(circle2)
    ax2.set_title("🌀 내부 자기장: 토로이달 성분 ($B_t$) 및 유체 변형 등고선", fontsize=12)
    fig.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.8)

    st.pyplot(fig)
    
    if st.session_state.running:
        time.sleep(0.05)
        st.rerun()

# ==========================================
# TAB 2: 알파 효과 & 오메가 효과 개별 시각화
# ==========================================
with tab2:
    st.subheader("🌪️ 다이나모 핵심 메커니즘 개별 시각화 룸")
    st.markdown("다이나모 이론을 지탱하는 두 가지 핵심 물리 현상이 자기력선을 어떻게 변형시키는지 개별적으로 관찰합니다.")

    st.sidebar.header("⚙️ 2. 메커니즘 시각화 설정")
    mech_type = st.sidebar.radio("관찰할 메커니즘 선택", ["오메가 효과 (차동 회전에 의한 감김)", "알파 효과 (코리올리 회오리에 의한 꼬임)"])
    distortion = st.sidebar.slider("변형 왜곡 정도 (Distortion Level)", 0.0, 4.0, 2.0, 0.2)

    # 기본 수직 자기력선 배치 (변형 전 상태 기본값)
    x_m = np.linspace(-2, 2, 30)
    y_m = np.linspace(-2, 2, 30)
    X_M, Y_M = np.meshgrid(x_m, y_m)
    
    fig_m, ax_m = plt.subplots(figsize=(7, 6))
    
    # 외핵 경계선
    circle_m = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='-', linewidth=2, label='Outer Core Boundary')
    ax_m.add_patch(circle_m)

    if mech_type == "오메가 효과 (차동 회전에 의한 감김)":
        st.markdown("""
        ### 🌀 오메가 효과 ($\Omega$-Effect)
        * **원리:** 지구가 자전할 때, 적도 부근과 극지방의 회전 속도가 서로 다른 **차동 회전(Differential Rotation)**이 발생합니다.
        * **현상:** 이 속도 차이로 인해 남북 방향의 자기력선(폴로이달)이 외핵 내부에 **동서 방향으로 강하게 감기며** 토로이달 자기장으로 증폭됩니다.
        """)
        
        # 차동 회전에 의한 속도 왜곡 장 유도
        # 중심부로 갈수록 회전 속도가 빨라지거나 달라짐을 모사
        U_x = -Y_M * np.exp(- (X_M**2 + Y_M**2)) * distortion
        U_y =  X_M * np.exp(- (X_M**2 + Y_M**2)) * distortion
        
        # 원래 남북 방향(Y축 방향)이었던 자기력선이 회전에 의해 감기는 모사
        B_x = U_x
        B_y = 1.0 + U_y
        
        # 시각화
        ax_m.streamplot(X_M, Y_M, B_x, B_y, color='crimson', density=1.2, linewidth=1.2, arrowstyle='->')
        ax_m.set_title("차동 회전에 의해 동서로 감기는 자기력선 ($\Omega$ 효과)", fontsize=11)

    else:
        st.markdown("""
        ### 🌪️ 알파 효과 ($\alpha$-Effect)
        * **원리:** 외핵 내부에서 열대류로 인해 액체 금속 유체가 상승하거나 하강할 때, 지구 자전에 의한 **코리올리 힘(Coriolis Force)**을 받습니다.
        * **현상:** 똑바로 가야 할 유체가 뱅글뱅글 꼬이면서 **3차원 회오리 소용돌이**를 만들고, 이 소용돌이가 감겨있던 자기력선을 꼬아주면서 다시 남북 방향의 폴로이달 자기장을 재생산합니다.
        """)
        
        # 국소적인 상승/하강 회오리 소용돌이 장 유도 (가운데 꼬임 발생)
        # 여러 소용돌이 셀 중 하나를 표현
        twist_mask = np.exp(-(X_M**2 + Y_M**2)/0.5)
        
        # 회오리 치며 꼬이는 벡터장
        B_x = 1.0 + distortion * (-Y_M * twist_mask)
        B_y = distortion * (X_M * twist_mask)
        
        # 시각화
        ax_m.streamplot(X_M, Y_M, B_x, B_y, color='royalblue', density=1.2, linewidth=1.2, arrowstyle='->')
        ax_m.set_title("코리올리 소용돌이에 의해 꼬이는 자기력선 ($\alpha$ 효과)", fontsize=11)

    ax_m.set_xlim(-2, 2)
    ax_m.set_ylim(-2, 2)
    ax_m.set_aspect('equal')
    ax_m.grid(True, linestyle=':', alpha=0.5)
    
    st.pyplot(fig_m)
