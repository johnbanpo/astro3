import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------------------------------------------
# 1. 페이지 설정 및 스타일 정의
# ----------------------------------------------------------------
st.set_page_config(page_title="Dynamo Theory Simulator", layout="wide")
st.title("🌋 다이나모 이론: 지구 자기장 생성 기전 시뮬레이터")
st.markdown("지구 외핵의 차동 회전($\\Omega$ 효과)과 코리올리 대류($\\alpha$ 효과)에 의한 자기장 유도를 시각화합니다.")
st.markdown("---")

# ----------------------------------------------------------------
# 2. 사이드바: 물리 및 시뮬레이션 파라미터 조절
# ----------------------------------------------------------------
st.sidebar.header("⚙️ 시뮬레이션 파라미터 설정")

# 물리적 계수 조절 
alpha_coeff = st.sidebar.slider("알파 효과 강도 (α-Effect)", 0.0, 5.0, 2.0, 0.1)
omega_coeff = st.sidebar.slider("오메가 효과 강도 (Ω-Effect)", 0.0, 10.0, 5.0, 0.5)
eta = st.sidebar.slider("자기 확산도 (Magnetic Diffusivity, η)", 0.1, 2.0, 0.5, 0.1)

# 시뮬레이션 제어 
grid_size = st.sidebar.slider("그리드 해상도 (Grid Size)", 20, 50, 30, 5)
dt = 0.01  # 타임 스텝 

# 애니메이션 제어 
run_simulation = st.sidebar.button("시뮬레이션 시작 / 재개")
reset_simulation = st.sidebar.button("초기화")

# ----------------------------------------------------------------
# 3. 세션 상태(Session State) 초기화 (캐싱 및 데이터 유지) 
# ----------------------------------------------------------------
if "step" not in st.session_state or reset_simulation:
    st.session_state.step = 0
    # 초기 폴로이달 자기장(Bp)과 토로이달 자기장(Bt) 생성
    X, Y = np.meshgrid(np.linspace(-2, 2, grid_size), np.linspace(-2, 2, grid_size))
    # 초기 조건: 부드러운 가우시안 형태의 쌍극자 자기장 모사
    st.session_state.Bp = np.exp(-(X**2 + Y**2)) 
    st.session_state.Bt = np.zeros_like(X)
    st.session_state.X = X
    st.session_state.Y = Y

# ----------------------------------------------------------------
# 4. 수치해석 엔진 (알파-오메가 다이나모 방정식 FDM 근사) 
# ----------------------------------------------------------------
def update_fields(Bp, Bt, alpha, omega, eta, dt):
    """
    간략화된 2D 다이나모 방정식의 유한차분법(FDM) 업데이트 
    dBt/dt = omega * dBp/dx + eta * Laplacian(Bt)
    dBp/dt = alpha * Bt + eta * Laplacian(Bp)
    """
    # 라플라시안(확산 항) 계산을 위한 패딩 연산 [cite: 8]
    def laplacian(Z):
        Z_top = np.roll(Z, -1, axis=0)
        Z_bottom = np.roll(Z, 1, axis=0)
        Z_left = np.roll(Z, -1, axis=1)
        Z_right = np.roll(Z, 1, axis=1)
        return Z_top + Z_bottom + Z_left + Z_right - 4 * Z

    # 1차 도함수 (오메가 효과 전단을 위함) [cite: 15]
    dBp_dx = (np.roll(Bp, -1, axis=1) - np.roll(Bp, 1, axis=1)) / 2.0

    # 방정식 적용 (Euler method) 
    lap_Bt = laplacian(Bt)
    lap_Bp = laplacian(Bp)
    
    # Bt는 Bp의 차동회전(오메가 효과)과 자기확산으로 변화 [cite: 15]
    new_Bt = Bt + dt * (omega * dBp_dx + eta * lap_Bt)
    # Bp는 Bt의 헬리시티 회오리(알파 효과)와 자기확산으로 변화 [cite: 19]
    new_Bp = Bp + dt * (alpha * Bt + eta * lap_Bp)
    
    return new_Bp, new_Bt

# ----------------------------------------------------------------
# 5. 메인 레이아웃 및 시각화 실시간 렌더링 
# ----------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌐 폴로이달 자기장 (Poloidal Field)")
    st.markdown("남북 방향의 주 자기장 성분입니다. 알파 효과에 의해 유지됩니다[cite: 19].")
    plot_holder_p = st.empty()

with col2:
    st.subheader("🌀 토로이달 자기장 (Toroidal Field)")
    st.markdown("외핵 내부에 갇힌 동서 방향 자기장입니다. 오메가 효과로 증폭됩니다[cite: 15].")
    plot_holder_t = st.empty()

status_text = st.empty()

# ----------------------------------------------------------------
# 6. 실시간 애니메이션 루프 
# ----------------------------------------------------------------
if run_simulation:
    while st.session_state.step < 200:  # 최대 200 타임스텝 진행
        # 필드 업데이트 
        st.session_state.Bp, st.session_state.Bt = update_fields(
            st.session_state.Bp, st.session_state.Bt, 
            alpha_coeff, omega_coeff, eta, dt
        )
        st.session_state.step += 1
        
        # 폴로이달 필드 시각화 (Matplotlib) 
        fig_p, ax_p = plt.subplots(figsize=(5, 4))
        cp = ax_p.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bp, cmap="RdBu_r", levels=20)
        fig_p.colorbar(cp, ax=ax_p)
        ax_p.set_title( generosity := f"Step: {st.session_state.step} | Poloidal (Bp)")
        plot_holder_p.pyplot(fig_p)
        plt.close(fig_p) # 메모리 누수 방지

        # 토로이달 필드 시각화 
        fig_t, ax_t = plt.subplots(figsize=(5, 4))
        ct = ax_t.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bt, cmap="inferno", levels=20)
        fig_t.colorbar(ct, ax=ax_t)
        ax_t.set_title(f"Step: {st.session_state.step} | Toroidal (Bt)")
        plot_holder_t.pyplot(fig_t)
        plt.close(fig_t) # 메모리 누수 방지

        # 상태 메시지 출력
        status_text.text(f"현재 시뮬레이션 진행 중... (Step {st.session_state.step}/200)")
        
        # 애니메이션 속도 조절 및 렉 방지 
        time.sleep(0.05)
else:
    status_text.text("시뮬레이션이 정지되었습니다. '시뮬레이션 시작'을 누르세요.")
