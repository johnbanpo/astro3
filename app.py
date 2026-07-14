
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 페이지 레이아웃 및 웹 디자인 정의
# ==========================================
st.set_page_config(
    page_title="지구 자기장 다이나모 시뮬레이터 Pro",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 다이나모 이론: 지구 자기장 생성 및 개별 메커니즘 시뮬레이터")
st.markdown("""
지구 외핵의 **차동 회전($\Omega$ 효과)**과 **코리올리 회오리 대류($\alpha$ 효과)**의 결합에 의한 
자기 유도(Magnetic Induction) 현상을 수치적으로 구현하고 개별 원리를 시각화하는 인터랙티브 연구 모델입니다.
""")
st.markdown("---")

# 탭 나누기 (통합 다이나모 시뮬레이션 / 메커니즘 개별 시각화)
tab1, tab2 = st.tabs(["📊 통합 다이나모 시뮬레이션", "🌪️ 메커니즘 개별 시각화 (α & Ω 효과)"])

# ==========================================
# 2. TAB 1: 통합 다이나모 시뮬레이션
# ==========================================
with tab1:
    st.sidebar.header("⚙️ 1. 통합 시뮬레이션 제어")
    
    # 물리 법칙 계수
    alpha_coeff = st.sidebar.slider(
        "알파 효과 강도 (α-Effect)", 0.0, 5.0, 2.5, 0.1, 
        help="토로이달 필드를 폴로이달 필드로 바꾸는 헬리시티 대류의 강도입니다.", key="alpha_t1"
    )
    omega_coeff = st.sidebar.slider(
        "오메가 효과 강도 (Ω-Effect)", 0.0, 10.0, 6.0, 0.5, 
        help="폴로이달 필드를 감아 늘려서 토로이달 필드를 만드는 차동 회전의 강도입니다.", key="omega_t1"
    )
    eta = st.sidebar.slider(
        "자기 확산도 (η)", 0.1, 2.0, 0.4, 0.1, 
        help="자기력선이 흩어지고 소멸하는 저항성 확산 계수입니다.", key="eta_t1"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎬 시뮬레이션 환경 제어")
    grid_size = st.sidebar.slider("그리드 해상도 (Grid)", 20, 50, 40, 5, key="grid_t1")
    dt = 0.005  # 수치적 안정성을 보장하는 고정 타임스텝

    # 애니메이션 플레이 상태 관리 변수 초기화
    if "running" not in st.session_state:
        st.session_state.running = False

    # 제어 버튼 레이아웃
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("▶️ 시뮬레이션 시작", key="start_t1"):
            st.session_state.running = True
    with c2:
        if st.button("⏸️ 일시정지", key="stop_t1"):
            st.session_state.running = False

    reset_btn = st.sidebar.button("🔄 시뮬레이션 초기화", key="reset_t1")

    # 데이터 및 경계 조건 세션 상태 초기화
    if "step" not in st.session_state or reset_btn:
        st.session_state.step = 0
        # 2D 좌표 평면 구축
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)
        st.session_state.X, st.session_state.Y = np.meshgrid(x, y)
        
        # 초기 조건: 지구 중심 쌍극자(Dipole)를 모사하는 평면 가우시안 폴로이달 자기장
        st.session_state.Bp = np.exp(-(st.session_state.X**2 + st.session_state.Y**2))
        st.session_state.Bt = np.zeros_like(st.session_state.X)
        st.session_state.running = False
        if reset_btn:
            st.rerun()

    # 수치해석 핵심 연산 엔진 (Laplacian Matrix 계산)
    def compute_next_step(Bp, Bt, alpha, omega, diffusivity, timestep):
        # 주기적 경계 조건을 적용한 공간 2차 미분 (라플라시안 - 확산 항)
        lap_Bp = (np.roll(Bp, -1, axis=0) + np.roll(Bp, 1, axis=0) +
                  np.roll(Bp, -1, axis=1) + np.roll(Bp, 1, axis=1) - 4 * Bp)
        
        lap_Bt = (np.roll(Bt, -1, axis=0) + np.roll(Bt, 1, axis=0) +
                  np.roll(Bt, -1, axis=1) + np.roll(Bt, 1, axis=1) - 4 * Bt)
        
        # 차동 회전에 의한 전단 응력 (x축 방향 1차 미분)
        dBp_dx = (np.roll(Bp, -1, axis=1) - np.roll(Bp, 1, axis=1)) / 2.0
        
        # 차분 방정식을 통한 시간 전진 연산 (Euler Method)
        saturation = 1.0 / (1.0 + 0.1 * Bt**2)
        
        next_Bt = Bt + timestep * (omega * dBp_dx + diffusivity * lap_Bt)
        next_Bp = Bp + timestep * (alpha * Bt * saturation + diffusivity * lap_Bp)
        
        return np.clip(next_Bp, -5.0, 5.0), np.clip(next_Bt, -5.0, 5.0)

    # 대시보드 렌더링 영역
    status_box = st.empty()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌐 주 자기장: 폴로이달 성분 ($B_p$) & 자기력선")
        st.caption("지구 내부에서 표면 밖으로 뻗어 나오는 실제 자기력선 경로가 실시간으로 중첩되어 나타납니다.")
        graph_p = st.empty()

    with col2:
        st.subheader("🌀 내부 자기장: 토로이달 성분 ($B_t$) & 변형 등고선")
        st.caption("외핵 내부에 갇혀 지구를 감싸고 도는 꼬인 자기장과 물리적 밀도 왜곡선을 보여줍니다.")
        graph_t = st.empty()

    # 자기력선 벡터 유도 함수
    def get_streamlines(Bp):
        # 자기장 강도의 경도를 사용하여 가상의 흐름 장(Streamline Vector) 계산
        dY, dX = np.gradient(Bp)
        # 90도 회전시켜 극을 감아 도는 자기력선 경로 구성
        return -dY, dX

    # 프레임 루프
    if st.session_state.running:
        while st.session_state.running and st.session_state.step < 1000:
            # 데이터 업데이트
            st.session_state.Bp, st.session_state.Bt = compute_next_step(
                st.session_state.Bp, st.session_state.Bt,
                alpha_coeff, omega_coeff, eta, dt
            )
            st.session_state.step += 1
            
            # 1) 폴로이달 필드 + 자기력선 그리기
            fig_p, ax_p = plt.subplots(figsize=(6, 4.5))
            contour_p = ax_p.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bp, cmap="RdBu_r", levels=25, vmin=-2, vmax=2)
            fig_p.colorbar(contour_p, ax=ax_p)
            
            # 실시간 자기력선 오버레이 (streamplot 적용)
            U, V = get_streamlines(st.session_state.Bp)
            ax_p.streamplot(st.session_state.X, st.session_state.Y, U, V, color='black', linewidth=0.8, density=0.9, arrowstyle='->')
            
            ax_p.set_title(f"Poloidal Intensity & Lines | Step: {st.session_state.step}")
            graph_p.pyplot(fig_p)
            
            # 2) 토로이달 필드 + 등고선 그리기
            fig_t, ax_t = plt.subplots(figsize=(6, 4.5))
            contour_t = ax_t.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bt, cmap="viridis", levels=25, vmin=-4, vmax=4)
            fig_t.colorbar(contour_t, ax=ax_t)
            
            # 유체 변형 등고선 레이어 추가
            ax_t.contour(st.session_state.X, st.session_state.Y, st.session_state.Bt, levels=10, colors='white', alpha=0.3, linewidths=0.8)
            
            ax_t.set_title(f"Toroidal Distortion | Step: {st.session_state.step}")
            graph_t.pyplot(fig_t)
            
            plt.close('all')
            status_box.info(f"🧬 다이나모 상호 순환 연산 진행 중... (Step: {st.session_state.step})")
            time.sleep(0.01)
    else:
        # 정지 상태 고정 화면
        fig_p, ax_p = plt.subplots(figsize=(6, 4.5))
        contour_p = ax_p.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bp, cmap="RdBu_r", levels=25, vmin=-2, vmax=2)
        fig_p.colorbar(contour_p, ax=ax_p)
        U, V = get_streamlines(st.session_state.Bp)
        ax_p.streamplot(st.session_state.X, st.session_state.Y, U, V, color='black', linewidth=0.8, density=0.9, arrowstyle='->')
        ax_p.set_title("Poloidal Field - Standing State")
        graph_p.pyplot(fig_p)
        
        fig_t, ax_t = plt.subplots(figsize=(6, 4.5))
        contour_t = ax_t.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bt, cmap="viridis", levels=25, vmin=-4, vmax=4)
        fig_t.colorbar(contour_t, ax=ax_t)
        ax_t.contour(st.session_state.X, st.session_state.Y, st.session_state.Bt, levels=10, colors='white', alpha=0.3, linewidths=0.8)
        ax_t.set_title("Toroidal Field - Standing State")
        graph_t.pyplot(fig_t)
        plt.close('all')
        
        status_box.warning(f"⏸️ 연산이 일시정지 되었습니다. (현재 연산 완료 단계: Step {st.session_state.step})")

# ==========================================
# 3. TAB 2: 알파 효과 & 오메가 효과 개별 시각화
# ==========================================
with tab2:
    st.subheader("🌪️ 다이나모 핵심 메커니즘 물리 분리 시각화 룸")
    st.markdown("통합 연산에서 벗어나, 자기력선이 유체의 움직임에 의해 '어떻게 왜곡되는지' 물리 현상만을 떼어내 시각적으로 학습합니다.")
    
    col_ctrl, col_plot = st.columns([1, 2])
    
    with col_ctrl:
        st.write("#### ⚙️ 시각화 파라미터 조절")
        mech_choice = st.radio(
            "시각화할 메커니즘을 선택하세요",
            ["🌀 오메가 효과 (Ω-Effect): 차동 회전에 의한 자기력선 감김", 
             "🌪️ 알파 효과 (α-Effect): 코리올리 회오리에 의한 자기력선 꼬임"]
        )
        distortion_val = st.slider("왜곡 강도 (Force Distortion)", 0.0, 4.0, 2.0, 0.2)
        
    with col_plot:
        # 가상의 물리 실험용 2D 그리드 구성
        g_size = 35
        x_m = np.linspace(-2.0, 2.0, g_size)
        y_m = np.linspace(-2.0, 2.0, g_size)
        X_M, Y_M = np.meshgrid(x_m, y_m)
        R_M = np.sqrt(X_M**2 + Y_M**2)
        
        fig_m, ax_m = plt.subplots(figsize=(7, 5.5))
        
        # 외핵의 가상 바운더리 라인
        outer_core = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', linewidth=2, label="Outer Core Boundary")
        ax_m.add_patch(outer_core)
        
        if "오메가 효과" in mech_choice:
            st.info("💡 **오메가 효과 원리:** 지구가 자전할 때 구형의 특성상 적도 부근의 자전 선속도가 극지방보다 빨라 회전 속도 차이가 발생합니다. 이로 인해 균일하게 남북 방향으로 뻗어 있던 원래 자기력선이 동서 방향으로 질질 끌려오며 가상의 실타래처럼 동그랗게 둘러 쌓이는 현상입니다.")
            
            # 수직 자기력선 방향 (남북 극지방 방향)
            B_x_init = np.zeros_like(X_M)
            B_y_init = np.ones_like(Y_M) * 1.0
            
            # 외핵 내부(R_M <= 1)에서만 작동하는 차동 자전 속도 기울기 시뮬레이션
            # 회전 전단 속도를 가우스 감쇠를 통해 내부 코어로 가중치 제한
            V_rotation_theta = Y_M * np.exp(-R_M**2) * distortion_val
            U_rotation_r = -X_M * np.exp(-R_M**2) * distortion_val
            
            # 원래 자기장에 차동 회전 성분이 더해짐 (수직선 -> 회전에 감김)
            B_x = B_x_init + V_rotation_theta
            B_y = B_y_init + U_rotation_r
            
            ax_m.streamplot(X_M, Y_M, B_x, B_y, color='crimson', density=1.1, linewidth=1.2, arrowstyle='->')
            ax_m.set_title("Ω-Effect: Differential Rotation Wrapping lines Toroidally", fontsize=11)
            
        else:
            st.info("💡 **알파 효과 원리:** 지구 외핵 중심의 열대류로 상승하는 액체 금속 유체가 자전에 의한 전향력(코리올리 효과)을 만나 3차원 회오리 소용돌이 기둥을 이룹니다. 이 기둥들이 수평하게 감겨있던 토로이달 자기선을 솟구치게 꼬면서, 다시 남북 방향(폴로이달 자기장)으로 재생산해 줍니다.")
            
            # 원래 동서 방향으로 감겨있던 수평 자기선 가정
            B_x_init = np.ones_like(X_M) * 1.0
            B_y_init = np.zeros_like(Y_M)
            
            # 상승 기류와 코리올리 회오리가 유발하는 비틀림 장 모사
            # 극소 헬리시티 셀 형태의 강한 소용돌이 배치
            helicity_mask = np.exp(-(X_M**2 + Y_M**2) / 0.6)
            
            # 소용돌이 힘에 의해 국소 부근에서 평면상 자기선이 수직 루프로 비틀어지는 모형 구현
            B_x = B_x_init - (Y_M * helicity_mask) * distortion_val
            B_y = B_y_init + (X_M * helicity_mask) * distortion_val
            
            ax_m.streamplot(X_M, Y_M, B_x, B_y, color='royalblue', density=1.1, linewidth=1.2, arrowstyle='->')
            ax_m.set_title("α-Effect: Helical Coriolis Convection twisting Field lines", fontsize=11)

        ax_m.set_xlim(-2.0, 2.0)
        ax_m.set_ylim(-2.0, 2.0)
        ax_m.set_aspect('equal')
        ax_m.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_m)
        plt.close(fig_m)
