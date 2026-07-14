import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ==========================================
# 1. 페이지 레이아웃 및 디자인 설정
# ==========================================
st.set_page_config(
    page_title="3D 지구 다이나모 시뮬레이터 V2",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 3D 지구 자기장 다이나모 이론 시뮬레이터 (V2)")
st.markdown("""
**외핵 대류($\\alpha$ 효과)**와 **차동 회전($\\Omega$ 효과)**이 유도하는 자기장 생성 기전을 
2D 유한차분법(FDM) 엔진으로 연산하고, 이를 **3D 지구 공간**에 실시간 입체 자력선으로 매핑합니다.
""")
st.markdown("---")

# ==========================================
# 2. 사이드바: 물리 파라미터 및 연산 제어
# ==========================================
st.sidebar.header("⚙️ 다이나모 물리 계수")
alpha_coeff = st.sidebar.slider("알파 효과 강도 (α - 헬리시티 대류)", 0.0, 5.0, 2.2, 0.1)
omega_coeff = st.sidebar.slider("오메가 효과 강도 (Ω - 차동 회전)", 0.0, 10.0, 6.5, 0.5)
eta = st.sidebar.slider("자기 확산도 (η - 저항 감쇄)", 0.05, 1.5, 0.35, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🚀 시뮬레이션 실행 제어")
grid_size = 40  # 2D 연산 그리드 해상도 고정 (안정성 확보)
dt = 0.005      # 시간 적분 간격
steps_per_click = st.sidebar.slider("1회 실행 당 연산 스텝 수", 10, 200, 50, 10)

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    step_button = st.button("▶️ 연산 진행", use_container_width=True)
with col_btn2:
    reset_button = st.button("🔄 초기화", use_container_width=True)

# ==========================================
# 3. 세션 상태(Session State) 초기화
# ==========================================
if "step" not in st.session_state or reset_button:
    st.session_state.step = 0
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    st.session_state.X, st.session_state.Y = np.meshgrid(x, y)
    
    # 초기 조건: 지구 중심 쌍극자(Dipole) 모사 가우시안 분포
    R2 = st.session_state.X**2 + st.session_state.Y**2
    st.session_state.Bp = np.exp(-R2)
    st.session_state.Bt = np.zeros_like(st.session_state.X)
    
    # 에너지 히스토리 저장용 리스트
    st.session_state.energy_history = []
    st.session_state.time_history = []
    if reset_button:
        st.rerun()

# ==========================================
# 4. 수치해석 고속 엔진 (벡터화 FDM)
# ==========================================
def solve_dynamo_step(Bp, Bt, alpha, omega, diffusivity, dt, steps):
    """지정된 스텝 수만큼 다이나모 방정식을 연속 적분합니다."""
    curr_Bp, curr_Bt = np.copy(Bp), np.copy(Bt)
    
    for _ in range(steps):
        # 2차 공간 미분 (Laplacian)
        lap_Bp = (np.roll(curr_Bp, -1, axis=0) + np.roll(curr_Bp, 1, axis=0) +
                  np.roll(curr_Bp, -1, axis=1) + np.roll(curr_Bp, 1, axis=1) - 4 * curr_Bp)
        lap_Bt = (np.roll(curr_Bt, -1, axis=0) + np.roll(curr_Bt, 1, axis=0) +
                  np.roll(curr_Bt, -1, axis=1) + np.roll(curr_Bt, 1, axis=1) - 4 * curr_Bt)
        
        # 차동 회전에 의한 전단력 (x방향 미분)
        dBp_dx = (np.roll(curr_Bp, -1, axis=1) - np.roll(curr_Bp, 1, axis=1)) / 2.0
        
        # 로렌츠 힘에 의한 다이나모 포화 제동 인자
        saturation = 1.0 / (1.0 + 0.1 * curr_Bt**2)
        
        # 시간 전진 (Euler Method)
        next_Bt = curr_Bt + dt * (omega * dBp_dx + diffusivity * lap_Bt)
        next_Bp = curr_Bp + dt * (alpha * curr_Bt * saturation + diffusivity * lap_Bp)
        
        # 수치적 발산 방지 클리핑
        curr_Bp = np.clip(next_Bp, -8.0, 8.0)
        curr_Bt = np.clip(next_Bt, -8.0, 8.0)
        
    return curr_Bp, curr_Bt

# 버튼 클릭 시 연산 수행
if step_button:
    st.session_state.Bp, st.session_state.Bt = solve_dynamo_step(
        st.session_state.Bp, st.session_state.Bt,
        alpha_coeff, omega_coeff, eta, dt, steps_per_click
    )
    st.session_state.step += steps_per_click

# 현재 상태의 총 자기 에너지 계산 및 기록
total_energy = np.sum(st.session_state.Bp**2 + st.session_state.Bt**2) * (4.0 / grid_size)**2
if len(st.session_state.time_history) == 0 or st.session_state.time_history[-1] != st.session_state.step:
    st.session_state.time_history.append(st.session_state.step)
    st.session_state.energy_history.append(total_energy)

# ==========================================
# 5. 3D 지구 자기력선 매핑 엔진 (Plotly 3D)
# ==========================================
def create_3d_dynamo_figure(Bp_field, Bt_field):
    fig = go.Figure()
    
    # 대표 자기장 강도 추출
    bp_max = np.max(np.abs(Bp_field)) + 1e-5
    bt_max = np.max(np.abs(Bt_field)) + 1e-5

    # [1] 3D 지구 구체 (Inner Core & Outer Core Boundary)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = 0.6 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 0.6 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 0.6 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale='Blues', opacity=0.3, showscale=False, name="외핵 경계"
    ))

    # [2] 폴로이달 자기력선 (3D 쌍극자 파라메트릭 곡선)
    # 극지방에서 뿜어져 나와 남북을 잇는 자력선 표현
    phi_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for phi in phi_angles:
        theta = np.linspace(0.1, np.pi-0.1, 50)
        # 자력선 방정식 r = L * sin^2(theta), L은 Bp 강도에 비례
        L = 1.2 * (1.0 + 0.3 * np.tanh(bp_max))
        r = L * (np.sin(theta)**2)
        
        xl = r * np.sin(theta) * np.cos(phi)
        yl = r * np.sin(theta) * np.sin(phi)
        zl = r * np.cos(theta)
        
        fig.add_trace(go.Scatter3d(
            x=xl, y=yl, z=zl, mode='lines',
            line=dict(color='cyan', width=3),
            opacity=0.8, name="폴로이달 자력선 (Bp)", showlegend=(phi==0)
        ))

    # [3] 토로이달 자기력선 (외핵 내부 동서 방향 꼬임 링)
    # 오메가 효과로 외핵 내부에 감긴 자력선 표현
    if bt_max > 0.1:
        z_rings = [-0.3, 0.0, 0.3]
        for z_val in z_rings:
            ring_phi = np.linspace(0, 2*np.pi, 60)
            ring_r = 0.45 + 0.1 * np.sin(3 * ring_phi) * (bt_max / 5.0) # 꼬임 왜곡 표현
            xr = ring_r * np.cos(ring_phi)
            yr = ring_r * np.sin(ring_phi)
            zr = np.full_like(ring_phi, z_val)
            
            fig.add_trace(go.Scatter3d(
                x=xr, y=yr, z=zr, mode='lines',
                line=dict(color='orange', width=5),
                name="토로이달 자력선 (Bt)", showlegend=(z_val==0.0)
            ))

    fig.update_layout(
        title="🌐 3D 지구 자기력선 공간 투영 (마우스로 드래그하여 회전 가능)",
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5], visible=False),
            yaxis=dict(range=[-1.5, 1.5], visible=False),
            zaxis=dict(range=[-1.5, 1.5], visible=False),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==========================================
# 6. 메인 화면 레이아웃 (3D 시각화 + 2D 단면)
# ==========================================
# 상단: 3D 지구 자기장 + 상태 요약
col_3d, col_info = st.columns([2, 1])

with col_3d:
    st.plotly_chart(
        create_3d_dynamo_figure(st.session_state.Bp, st.session_state.Bt),
        use_container_width=True
    )

with col_info:
    st.subheader("📊 다이나모 상태 모니터")
    st.metric("현재 연산 스텝", f"{st.session_state.step} Step")
    st.metric("총 자기장 에너지", f"{total_energy:.4f}", delta=f"{total_energy - st.session_state.energy_history[0]:.4f}")
    
    # 자기 에너지 시계열 차트
    if len(st.session_state.energy_history) > 1:
        df_energy = pd.DataFrame({
            "Step": st.session_state.time_history,
            "Magnetic Energy": st.session_state.energy_history
        })
        st.line_chart(df_energy, x="Step", y="Magnetic Energy", height=250)
    else:
        st.info("좌측 사이드바의 [▶️ 연산 진행] 버튼을 눌러 에너지를 축적하세요.")

st.markdown("---")

# 하단: 외핵 내부 2D 단면 열지도 (Heatmap)
st.subheader("🔍 외핵 내부 자기장 2D 단면도 (FDM Solver Core)")
col_p, col_t = st.columns(2)

with col_p:
    fig_bp = go.Figure(data=go.Heatmap(
        z=st.session_state.Bp, x=x, y=y, colorscale='RdBu', zmid=0,
        colorbar=dict(title="Bp 강도")
    ))
    fig_bp.update_layout(title="폴로이달 자기장 단면 (남북 방향)", height=350, margin=dict(l=10, r=10, b=10, t=30))
    st.plotly_chart(fig_bp, use_container_width=True)

with col_t:
    fig_bt = go.Figure(data=go.Heatmap(
        z=st.session_state.Bt, x=x, y=y, colorscale='Solar', zmid=0,
        colorbar=dict(title="Bt 강도")
    ))
    fig_bt.update_layout(title="토로이달 자기장 단면 (동서 방향 꼬임)", height=350, margin=dict(l=10, r=10, b=10, t=30))
    st.plotly_chart(fig_bt, use_container_width=True)
