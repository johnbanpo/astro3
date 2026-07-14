import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 페이지 레이아웃 및 웹 디자인 정의
# ==========================================
st.set_page_config(
    page_title="지구 자기장 다이나모 시뮬레이터",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 다이나모 이론: 지구 자기장 생성 기전 시뮬레이터")
st.markdown("""
지구 외핵의 **차동 회전($\Omega$ 효과)**과 **코리올리 회오리 대류($\alpha$ 효과)**의 결합에 의한 
자기 유도(Magnetic Induction) 현상을 수치적으로 구현한 인터랙티브 연구 모델입니다.
""")
st.markdown("---")

# ==========================================
# 2. 사이드바 제어 패널 (파라미터 설정)
# ==========================================
st.sidebar.header("⚙️ 연구용 물리 계수 제어")

# 물리 법칙 계수
alpha_coeff = st.sidebar.slider("알파 효과 강도 (α-Effect)", 0.0, 5.0, 2.5, 0.1, help="토로이달 필드를 폴로이달 필드로 바꾸는 헬리시티 대류의 강도입니다.")
omega_coeff = st.sidebar.slider("오메가 효과 강도 (Ω-Effect)", 0.0, 10.0, 6.0, 0.5, help="폴로이달 필드를 감아 늘려서 토로이달 필드를 만드는 차동 회전의 강도입니다.")
eta = st.sidebar.slider("자기 확산도 (Magnetic Diffusivity, η)", 0.1, 2.0, 0.4, 0.1, help="자기력선이 흩어지고 소멸하는 저항성 확산 계수입니다.")

st.sidebar.markdown("---")
st.sidebar.header("🎬 시뮬레이션 환경 제어")
grid_size = st.sidebar.slider("그리드 해상도 (Grid)", 20, 50, 40, 5)
dt = 0.005  # 수치적 안정성을 보장하는 고정 타임스텝

# 애니메이션 플레이 상태 관리 변수 초기화
if "running" not in st.session_state:
    st.session_state.running = False

# 제어 버튼 레이아웃
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("▶️ 시뮬레이션 시작"):
        st.session_state.running = True
with c2:
    if st.button("⏸️ 일시정지"):
        st.session_state.running = False

reset_btn = st.sidebar.button("🔄 시뮬레이션 초기화")

# ==========================================
# 3. 데이터 및 경계 조건 세션 상태 초기화
# ==========================================
if "step" not in st.session_state or reset_btn:
    st.session_state.step = 0
    # 2D 좌표 평면 구축
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    st.session_state.X, st.session_state.Y = np.meshgrid(x, y)
    
    # 초기 조건: 지구 중심 쌍극자(Dipole)를 모사하는 평면 가우시안 폴로이달 자기장
    st.session_state.Bp = np.exp(-(st.session_state.X**2 + st.session_state.Y**2))
    # 초기 토로이달 자기장은 유도되기 전이므로 0으로 설정
    st.session_state.Bt = np.zeros_like(st.session_state.X)
    st.session_state.running = False
    if reset_btn:
        st.rerun()

# ==========================================
# 4. 수치해석 핵심 연산 엔진 (Laplacian Matrix 계산)
# ==========================================
def compute_next_step(Bp, Bt, alpha, omega, diffusivity, timestep):
    """
    유한차분법(FDM) 기반의 2D 알파-오메가 다이나모 유도 방정식 솔버.
    주기적 경계 조건(np.roll)을 활용해 계산 속도를 최적화하고 에러를 차단합니다.
    """
    # 주기적 경계 조건을 적용한 공간 2차 미분 (라플라시안 - 확산 항)
    lap_Bp = (np.roll(Bp, -1, axis=0) + np.roll(Bp, 1, axis=0) +
              np.roll(Bp, -1, axis=1) + np.roll(Bp, 1, axis=1) - 4 * Bp)
    
    lap_Bt = (np.roll(Bt, -1, axis=0) + np.roll(Bt, 1, axis=0) +
              np.roll(Bt, -1, axis=1) + np.roll(Bt, 1, axis=1) - 4 * Bt)
    
    # 차동 회전에 의한 전단 응력 (x축 방향 1차 미분)
    dBp_dx = (np.roll(Bp, -1, axis=1) - np.roll(Bp, 1, axis=1)) / 2.0
    
    # 차분 방정식을 통한 시간 전진 연산 (Euler Method)
    # 로렌츠 힘 피드백에 의한 무한 발산 방지를 위해 탄탄한 포화 메커니즘을 상수로 제어
    saturation = 1.0 / (1.0 + 0.1 * Bt**2)
    
    next_Bt = Bt + timestep * (omega * dBp_dx + diffusivity * lap_Bt)
    next_Bp = Bp + timestep * (alpha * Bt * saturation + diffusivity * lap_Bp)
    
    # 수치적 오버플로우로 인한 깨짐 방지 안전장치
    return np.clip(next_Bp, -5.0, 5.0), np.clip(next_Bt, -5.0, 5.0)

# ==========================================
# 5. 메인 대시보드 렌더링 영역
# ==========================================
status_box = st.empty()
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌐 주 자기장: 폴로이달 성분 (Poloidal Field - Bp)")
    st.caption("남북 방향으로 형성되는 겉보기 자기장입니다. 알파 효과로 유지됩니다.")
    graph_p = st.empty()

with col2:
    st.subheader("🌀 내부 자기장: 토로이달 성분 (Toroidal Field - Bt)")
    st.caption("외핵 내부에 갇혀 지구를 감싸고 도는 자기장입니다. 오메가 효과로 증폭됩니다.")
    graph_t = st.empty()

# ==========================================
# 6. 고속 프레임 재생 애니메이션 루프
# ==========================================
if st.session_state.running:
    # 실시간 프레임 드롭을 줄이기 위해 컨텍스트 분리
    while st.session_state.running and st.session_state.step < 1000:
        # 데이터 업데이트
        st.session_state.Bp, st.session_state.Bt = compute_next_step(
            st.session_state.Bp, st.session_state.Bt,
            alpha_coeff, omega_coeff, eta, dt
        )
        st.session_state.step += 1
        
        # 1) 폴로이달 필드 그리기
        fig_p, ax_p = plt.subplots(figsize=(6, 4.5))
        contour_p = ax_p.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bp, cmap="RdBu_r", levels=25, vmin=-2, vmax=2)
        fig_p.colorbar(contour_p, ax=ax_p)
        ax_p.set_title(f"Poloidal Intensity | Step: {st.session_state.step}")
        graph_p.pyplot(fig_p)
        
        # 2) 토로이달 필드 그리기
        fig_t, ax_t = plt.subplots(figsize=(6, 4.5))
        contour_t = ax_t.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bt, cmap="viridis", levels=25, vmin=-4, vmax=4)
        fig_t.colorbar(contour_t, ax=ax_t)
        ax_t.set_title(f"Toroidal Distortion | Step: {st.session_state.step}")
        graph_t.pyplot(fig_t)
        
        # 가비지 컬렉션을 통한 메모리 폭발 누수 차단 (가장 중요)
        plt.close('all')
        
        status_box.info(f"🧬 다이나모 상호 순환 연산 진행 중... (Step: {st.session_state.step})")
        
        # Streamlit 프레임 렌더링 동기화를 위한 찰나의 대기시간
        time.sleep(0.01)
else:
    # 정지 상태이거나 초기에 정적 화면 고정
    fig_p, ax_p = plt.subplots(figsize=(6, 4.5))
    contour_p = ax_p.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bp, cmap="RdBu_r", levels=25, vmin=-2, vmax=2)
    fig_p.colorbar(contour_p, ax=ax_p)
    ax_p.set_title("Poloidal Field - Standing State")
    graph_p.pyplot(fig_p)
    
    fig_t, ax_t = plt.subplots(figsize=(6, 4.5))
    contour_t = ax_t.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bt, cmap="viridis", levels=25, vmin=-4, vmax=4)
    fig_t.colorbar(contour_t, ax=ax_t)
    ax_t.set_title("Toroidal Field - Standing State")
    graph_t.pyplot(fig_t)
    plt.close('all')
    
    status_box.warning(f"⏸️ 연산이 일시정지 되었습니다. (현재 연산 완료 단계: Step {st.session_state.step})")
