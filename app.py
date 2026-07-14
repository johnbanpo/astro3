import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 페이지 기본 설정 및 사용자 정의 CSS
# ==========================================
st.set_page_config(
    page_title="지구 다이나모 이론 시뮬레이터 Pro",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 다크 모드 UI 최적화 및 스타일링
st.markdown("""
<style>
.main {background-color: #0e1117; color: #ffffff;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    height: 50px; 
    white-space: pre-wrap; 
    background-color: #1e222b; 
    border-radius: 4px; 
    color: #ffffff; 
    padding-left: 16px; 
    padding-right: 16px;
}
.stTabs [aria-selected="true"] {background-color: #ff4b4b !important; font-weight: bold;}
div.stButton > button:first-child {
    background-color: #ff4b4b; 
    color: white; 
    border: none; 
    border-radius: 5px; 
    padding: 0.5em 1em; 
    font-weight: bold; 
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# 애플리케이션 타이틀 정의
st.title("🌍 지구 다이나모 이론 (Geodynamo Theory) 시뮬레이터 Pro")
st.subheader("지구 내부의 유체 운동이 어떻게 거대한 우주적 자석을 만드는가?")
st.markdown("---")

# ==========================================
# 2. 사이드바 제어 패널 (변수명 오류 완벽 수정)
# ==========================================
st.sidebar.header("⚙️ 다이나모 매개변수 제어")
st.sidebar.markdown("이 변수들이 지구 외핵 내의 발전기(Dynamo) 작용을 결정합니다.")

# 🚨 [버그 수정]: 아래 연산엔진 호출 변수명과 100% 일치시킴
alpha_coeff = st.sidebar.slider(
    "알파(α) 효과 강도 (Helicity)", 
    min_value=0.0, max_value=5.0, value=2.5, step=0.1,
    help="외핵 유체의 대류와 코리올리 효과로 인해 자기력선이 꼬이면서 폴로이달 자기장을 생성하는 효율입니다."
)
omega_coeff = st.sidebar.slider(
    "오메가(Ω) 효과 강도 (Differential Rotation)", 
    min_value=0.0, max_value=15.0, value=8.0, step=0.5,
    help="지구 자전 속도의 차이(차등 회전)로 인해 폴로이달 자기장을 끌어당겨 동서 방향의 토로이달 자기장으로 변환하는 힘입니다."
)
eta = st.sidebar.slider(
    "자기 확산도 (Magnetic Diffusivity, η)", 
    min_value=0.1, max_value=3.0, value=0.4, step=0.1,
    help="전도성 유체 내에서 자기장이 소멸하고 흩어지는 저항성 분산 비율입니다. 값이 너무 크면 다이나모가 붕괴합니다."
)

st.sidebar.markdown("---")
st.sidebar.header("🎬 시뮬레이션 환경 제어")
grid_size = st.sidebar.slider("그리드 해상도 (Grid)", 20, 50, 40, 5)
max_steps = st.sidebar.slider("최대 시뮬레이션 스텝 (Max Steps)", min_value=100, max_value=1000, value=500, step=50)
dt = 0.005  # 수치적 안정성을 보장하는 유한차분 타임스텝

# 애니메이션 플레이 상태 변수 제어
if "running" not in st.session_state:
    st.session_state.running = False

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
if "step" not in st.session_state or reset_btn or "X" not in st.session_state or st.session_state.X.shape[0] != grid_size:
    st.session_state.step = 0
    x_coord = np.linspace(-2, 2, grid_size)
    y_coord = np.linspace(-2, 2, grid_size)
    st.session_state.X, st.session_state.Y = np.meshgrid(x_coord, y_coord)
    
    # 초기 조건: 지구 중심 쌍극자를 모사하는 가우시안 필드
    st.session_state.Bp = np.exp(-(st.session_state.X**2 + st.session_state.Y**2))
    st.session_state.Bt = np.zeros_like(st.session_state.X)
    
    st.session_state.history_time = []
    st.session_state.history_Bp_energy = []
    st.session_state.history_Bt_energy = []
    st.session_state.running = False
    if reset_btn:
        st.rerun()

# ==========================================
# 4. 수치해석 핵심 연산 엔진 (2D 유한차분법 Solver)
# ==========================================
def compute_next_step(Bp, Bt, alpha, omega, diffusivity, timestep):
    lap_Bp = (np.roll(Bp, -1, axis=0) + np.roll(Bp, 1, axis=0) + np.roll(Bp, -1, axis=1) + np.roll(Bp, 1, axis=1) - 4 * Bp)
    lap_Bt = (np.roll(Bt, -1, axis=0) + np.roll(Bt, 1, axis=0) + np.roll(Bt, -1, axis=1) + np.roll(Bt, 1, axis=1) - 4 * Bt)
    
    dBp_dx = (np.roll(Bp, -1, axis=1) - np.roll(Bp, 1, axis=1)) / 2.0
    saturation = 1.0 / (1.0 + 0.1 * Bt**2)
    
    next_Bt = Bt + timestep * (omega * dBp_dx + diffusivity * lap_Bt)
    next_Bp = Bp + timestep * (alpha * Bt * saturation + diffusivity * lap_Bp)
    return np.clip(next_Bp, -5.0, 5.0), np.clip(next_Bt, -5.0, 5.0)

# ==========================================
# 5. 메인 탭 레이아웃 구성
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📖 다이나모 이론 학습", 
    "📊 실시간 2D 수치 시뮬레이션", 
    "🌐 3D 지구 자기력선 시각화", 
    "🌪️ 핵심 메커니즘 개별 실험실"
])

# -------------------------------------------------------------------------
# TAB 1: 다이나모 이론 학습
# -------------------------------------------------------------------------
with tab1:
    st.header("🧠 다이나모 이론 (Geodynamo Theory) 이란?")
    col_theory_l, col_theory_r = st.columns([3, 2])
    with col_theory_l:
        st.markdown("""
        지구의 자기장은 단순히 자성을 띤 거대한 광물이 안에 들어있기 때문이 아닙니다. 지구 내부 깊은 곳(외핵)은 **섭씨 4,000도가 넘는 액체 금속(철과 니켈)**으로 가득 차 있으며, 이 액체 금속이 끊임없이 흐르며 전류를 만들어내고, 그 전류가 다시 자기장을 만드는 **'자가 발전기(Self-exciting Dynamo)'** 구조를 이루고 있습니다. 이를 **다이나모 이론**이라고 합니다.
        
        다이나모가 지속적으로 작동하기 위해서는 다음과 같은 세 가지 조건이 필요합니다.
        1. **도체인 유체**: 전류를 흘릴 수 있는 외핵의 액체 철.
        2. **에너지원**: 외핵 상하부의 온도 차이로 인해 생기는 **열대류** 및 화학적 성분 차이로 인한 **조성 대류**.
        3. **회전력**: 지구 자전으로 인해 유체 흐름을 비틀어주는 **코리올리 힘(Coriolis force)**.
        """)
    with col_theory_r:
        st.info("💡 **핵심 직관**: 전기 모터에 전기를 넣으면 회전하듯, 반대로 유체의 회전 운동을 자기장 영역에 넣으면 전기가 만들어지고 자기장이 증폭됩니다.")
    
    st.markdown("---")
    st.subheader("🔄 다이나모의 두 가지 기둥: 폴로이달과 토로이달 성분")
    col_pol, col_tor = st.columns(2)
    with col_pol:
        st.markdown("#### 1. 폴로이달 성분 (Poloidal Field, $B_p$)")
        st.markdown("""
        * **형태**: 자석의 북극과 남극을 잇는 전형적인 **남북 방향의 자기장**입니다. 지구 외부로 뻗어 나와 나침반을 움직이게 만드는 자기장이 바로 이 성분입니다.
        * **역할**: 오메가($\Omega$) 효과의 원료가 됩니다.
        """)
    with col_tor:
        st.markdown("#### 2. 토로이달 성분 (Toroidal Field, $B_t$)")
        st.markdown("""
        * **형태**: 지구 자전축을 둥글게 감싸는 **동서 방향의 도넛 모양 자기장**입니다. 외핵 내부에 갇혀 있어 지구 표면에서는 직접 관측되지 않습니다.
        * **역할**: 알파($\\alpha$) 효과의 원료가 됩니다.
        """)
        
    st.markdown("---")
    st.subheader("⚙️ 순환 메커니즘: $\\alpha$ 효과와 $\\Omega$ 효과 공식")
    col_alpha, col_omega = st.columns(2)
    with col_omega:
        st.markdown("### 🌀 오메가 효과 ($\Omega$-Effect) : $B_p \\rightarrow B_t$")
        st.markdown("""
        * **원리**: 지구 자전 시 적도 부근과 극 부근의 회전 속도가 다른 **차동 회전(Differential Rotation)**이 발생합니다.
        * **현상**: 이 속도 차이로 인해 남북 방향의 폴로이달 자기선이 동서 방향으로 팽팽하게 감기면서 강한 **토로이달 자기장**으로 변환됩니다.
        """)
        st.latex(r"\frac{\partial B_t}{\partial t} \approx \Delta \Omega \cdot B_p + \eta \nabla^2 B_t")
    with col_alpha:
        st.markdown("### 🌪️ 알파 효과 ($\\alpha$-Effect) : $B_t \\rightarrow B_p$")
        st.markdown("""
        * **원리**: 뜨거운 외핵 유체가 상승할 때, 지구 자전에 의한 **코리올리 힘**을 받아 소용돌이치며 회전 상승합니다(나선형 유동, Helicity).
        * **현상**: 이 나선형 흐름이 동서 방향의 토로이달 자기선을 잡아서 고리 모양으로 비틀어 올립니다. 수많은 미세 소용돌이가 만든 고리들이 합쳐져 다시 거대한 **폴로이달 자기장**을 재생산합니다.
        """)
        st.latex(r"\frac{\partial B_p}{\partial t} \approx \alpha \cdot B_t + \eta \nabla^2 B_p")
    st.success("✅ **결론**: 폴로이달 자기장이 $\\Omega$ 효과로 토로이달이 되고, 토로이달 자기장이 $\\alpha$ 효과로 다시 폴로이달이 되는 이 순환 고리($\\alpha\\Omega$-Dynamo) 덕분에 지구 자기장은 수십억 년 동안 꺼지지 않고 유지될 수 있습니다.")

# -------------------------------------------------------------------------
# TAB 2: 실시간 2D 수치 시뮬레이션
# -------------------------------------------------------------------------
with tab2:
    st.header("📊 $\\alpha\\Omega$-다이나모 2D 고해상도 시뮬레이션")
    st.markdown("사이드바의 제어 패널을 통해 유체 유도 계수를 변경하고 실시간 물리 연산 흐름을 관찰하세요.")
    status_box = st.empty()
    
    col_graph1, col_graph2 = st.columns(2)
    with col_graph1:
        st.subheader("🌐 주 자기장: 폴로이달 성분 ($B_p$)")
        graph_p = st.empty()
    with col_graph2:
        st.subheader("🌀 내부 자기장: 토로이달 성분 ($B_t$)")
        graph_t = st.empty()
        
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("📈 시간 흐름에 따른 자기장 에너지 변화 (Plotly)")
        chart_energy = st.empty()
    with col_chart2:
        st.subheader("📐 핵 반경 방향에 따른 최종 자기장 분포 (Plotly)")
        chart_profile = st.empty()

    # 애니메이션 실행 루프
    if st.session_state.running:
        while st.session_state.running and st.session_state.step < max_steps:
            st.session_state.Bp, st.session_state.Bt = compute_next_step(
                st.session_state.Bp, st.session_state.Bt, alpha_coeff, omega_coeff, eta, dt
            )
            st.session_state.step += 1
            
            st.session_state.history_time.append(st.session_state.step * dt)
            st.session_state.history_Bp_energy.append(np.sum(st.session_state.Bp**2))
            st.session_state.history_Bt_energy.append(np.sum(st.session_state.Bt**2))
            
            # 2D 폴로이달 시각화
            fig_p, ax_p = plt.subplots(figsize=(6, 4.5))
            contour_p = ax_p.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bp, cmap="RdBu_r", levels=25, vmin=-2, vmax=2)
            fig_p.colorbar(contour_p, ax=ax_p)
            ax_p.set_title(f"Poloidal Field | Step: {st.session_state.step}")
            graph_p.pyplot(fig_p)
            
            # 2D 토로이달 시각화
            fig_t, ax_t = plt.subplots(figsize=(6, 4.5))
            contour_t = ax_t.contourf(st.session_state.X, st.session_state.Y, st.session_state.Bt, cmap="viridis", levels=25, vmin=-4, vmax=4)
            fig_t.colorbar(contour_t, ax=ax_t)
            ax_t.set_title(f"Toroidal Field | Step: {st.session_state.step}")
            graph_t.pyplot(fig_t)
            
            # Plotly 에너지 차트
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(x=st.session_state.history_time, y=st.session_state.history_Bp_energy, name="폴로이달 에너지 ($B_p^2$)", line=dict(color='#ff4b4b', width=2.5)))
            fig_energy.add_trace(go.Scatter(x=st.session_state.history_time, y=st.session_state.history_Bt_energy, name="토로이달 에너지 ($B_t^2$)", line=dict(color='#00f0ff', width=2.5)))
            fig_energy.update_layout(template="plotly_dark", xaxis_title="무차원 시간 (Time)", yaxis_title="자기장 에너지", margin=dict(l=20, r=20, t=20, b=20))
            chart_energy.plotly_chart(fig_energy, use_container_width=True)
            
            # Plotly 프로파일 차트
            center_idx = grid_size // 2
            r_grid = np.linspace(0.3, 1.0, grid_size - center_idx)
            fig_profile = go.Figure()
            fig_profile.add_trace(go.Scatter(x=r_grid, y=st.session_state.Bp[center_idx, center_idx:], name="폴로이달 성분 ($B_p$)", line=dict(color='#ff4b4b', dash='dash')))
            fig_profile.add_trace(go.Scatter(x=r_grid, y=st.session_state.Bt[center_idx, center_idx:], name="토로이달 성분 ($B_t$)", line=dict(color='#00f0ff')))
            fig_profile.update_layout(template="plotly_dark", xaxis_title="외핵 내 규격화 반경 (r)", yaxis_title="자기장 세기", margin=dict(l=20, r=20, t=20, b=20))
            chart_profile.plotly_chart(fig_profile, use_container_width=True)
            
            plt.close('all')
            status_box.info(f"🧬 다이나모 상호 순환 연산 진행 중... (Step: {st.session_state.step}/{max_steps})")
            time.sleep(0.01)
            
        if st.session_state.step >= max_steps:
            st.session_state.running = False
            st.rerun()
    else:
        # 정지 상태 고정 화면 출력
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
        
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(x=st.session_state.history_time, y=st.session_state.history_Bp_energy, name="폴로이달 에너지 ($B_p^2$)", line=dict(color='#ff4b4b', width=2.5)))
        fig_energy.add_trace(go.Scatter(x=st.session_state.history_time, y=st.session_state.history_Bt_energy, name="토로이달 에너지 ($B_t^2$)", line=dict(color='#00f0ff', width=2.5)))
        fig_energy.update_layout(template="plotly_dark", xaxis_title="무차원 시간 (Time)", yaxis_title="자기장 에너지", margin=dict(l=20, r=20, t=20, b=20))
        chart_energy.plotly_chart(fig_energy, use_container_width=True)
        
        center_idx = grid_size // 2
        r_grid = np.linspace(0.3, 1.0, grid_size - center_idx)
        fig_profile = go.Figure()
        fig_profile.add_trace(go.Scatter(x=r_grid, y=st.session_state.Bp[center_idx, center_idx:], name="폴로이달 성분 ($B_p$)", line=dict(color='#ff4b4b', dash='dash')))
        fig_profile.add_trace(go.Scatter(x=r_grid, y=st.session_state.Bt[center_idx, center_idx:], name="토로이달 성분 ($B_t$)", line=dict(color='#00f0ff')))
        fig_profile.update_layout(template="plotly_dark", xaxis_title="외핵 내 규격화 반경 (r)", yaxis_title="자기장 세기", margin=dict(l=20, r=20, t=20, b=20))
        chart_profile.plotly_chart(fig_profile, use_container_width=True)
        plt.close('all')
        
        if len(st.session_state.history_Bp_energy) > 0:
            final_eng = st.session_state.history_Bp_energy[-1]
            if final_eng < 0.01:
                st.error("📉 **다이나모 소멸 (Decay)**: 자기 확산도가 생산력보다 강해 자기장이 유지되지 못하고 소멸했습니다.")
            elif final_eng > 15.0:
                st.warning("📈 **폭발적 발산 (Unbounded Growth)**: 선형 수치 방정식 특성상 한계 임계치를 초과하여 발산 중입니다.")
            else:
                st.success("🔄 **지속 가능한 진동형 다이나모 (Self-Sustaining)**: 알파와 오메가 효과가 확산과 평형을 이루어 스스로 자기장을 유지합니다.")
        else:
            status_box.warning(f"⏸️ 연산이 대기 중입니다. (현재 연산 완료 단계: Step {st.session_state.step})")

# -------------------------------------------------------------------------
# TAB 3: 3D 지구 자기력선 시각화룸 (웅장하게 크기 및 시인성 개선)
# -------------------------------------------------------------------------
with tab3:
    st.header("🌐 3차원 지구 모형과 Plotly 입체 우주 자기력선")
    st.markdown("현재 폴로이달 자기장의 연산 강도를 기반으로 도출된 3D 다이폴 구조입니다. 마우스 드래그로 회전 및 확대 조작이 가능합니다.")
    
    # 💡 [3D 시인성 대폭 개선]: 선이 작아지거나 지구 속에 파묻히지 않도록 기본 스케일 하한선 대폭 증폭
    def get_dipole_lines(strength, num_lines=24, num_points=60):
        lines = []
        # 최소 반경 스케일을 1.2로 끌어올려 우주 공간 보호막 형태가 뚜렷이 보이도록 보정
        effective_scale = 1.2 + np.clip(strength * 2.5, 0.0, 3.5)
        longitudes = np.linspace(0, 2 * np.pi, num_lines, endpoint=False)
        loop_sizes = [0.8, 1.4, 2.1, 2.9] # 다이폴 고리 반경 간격 확장
        
        for lon in longitudes:
            for base_r in loop_sizes:
                r_max = base_r * effective_scale
                min_val = np.clip(0.5 / r_max, 0, 1)
                theta_min = np.arcsin(np.sqrt(min_val))
                theta = np.linspace(theta_min, np.pi - theta_min, num_points)
                r = r_max * (np.sin(theta) ** 2)
                
                x = r * np.sin(theta) * np.cos(lon)
                y = r * np.sin(theta) * np.sin(lon)
                z = r * np.cos(theta)
                lines.append((x, y, z))
        return lines

    current_strength = np.mean(np.abs(st.session_state.Bp))
    field_lines = get_dipole_lines(current_strength)
    fig_3d = go.Figure()
    
    # 투명한 지구 표면 구체 드로잉
    u_s = np.linspace(0, 2 * np.pi, 30)
    v_s = np.linspace(0, np.pi, 30)
    xs = 0.5 * np.outer(np.cos(u_s), np.sin(v_s))
    ys = 0.5 * np.outer(np.sin(u_s), np.sin(v_s))
    zs = 0.5 * np.outer(np.ones(np.size(u_s)), np.cos(v_s))
    fig_3d.add_trace(go.Surface(
        x=xs, y=ys, z=zs, 
        colorscale=[[0, '#104e8b'], [0.5, '#20639b'], [1, '#3caea3']], 
        showscale=False, opacity=0.45, name="지구 표면"
    ))
    
    # 내부 쌍극자 막대자석 축 오버레이 (N: 빨강 / S: 파랑 선명하게 강조)
    fig_3d.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 0.8], mode='lines', line=dict(color='#ff0000', width=9), name="막대자석 코어 (N극)"))
    fig_3d.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-0.8, 0], mode='lines', line=dict(color='#0000ff', width=9), name="막대자석 코어 (S극)"))
    
    # 계산된 우주 자기선 고리 매핑 (네온 블루-화이트로 시인성 확보)
    for idx, (x_l, y_l, z_l) in enumerate(field_lines):
        fig_3d.add_trace(go.Scatter3d(
            x=x_l, y=y_l, z=z_l, mode='lines', 
            line=dict(color='rgba(0, 255, 255, 0.65)', width=3.0),
            showlegend=True if idx == 0 else False, name="우주 자기력선 보호막"
        ))
        
    fig_3d.update_layout(
        template="plotly_dark", 
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(eye=dict(x=2.0, y=2.0, z=1.4))
        ),
        height=650
    )
    
    col_3d_l, col_3d_r = st.columns([3, 1])
    with col_3d_l:
        st.plotly_chart(fig_3d, use_container_width=True)
    with col_3d_r:
        st.markdown("### 💡 3D 자기장 인터랙티브 피드백")
        st.markdown(f"**현재 다이나모 유도강도 지수**: `{current_strength:.4f}`")
        st.markdown("""
        * **자기장의 팽창과 수축**: 수치 시뮬레이션 탭에서 연산을 재개하여 에너지가 커지면, 이곳 3D 궤적 내 하늘색 보호막의 볼륨감이 실시간 비례 연동하여 커집니다.
        * **자석 유도축 고정**: 구체 상하단의 굵은 적색/청색 축선은 행성 내핵-외핵 중심의 쌍극 발전 극성을 상징합니다.
        """)

# -------------------------------------------------------------------------
# TAB 4: 핵심 메커니즘 개별 실험실
# -------------------------------------------------------------------------
with tab4:
    st.header("🌪️ 다이나모 핵심 물리 변형 분리 학습실")
    st.markdown("통합 다이나모 수치 해석과 별개로, 유체의 미시적 거동이 자기선 지오메트리를 어떻게 꼬아놓는지 개별 법칙만 분리해 실험합니다.")
    
    col_lab_l, col_lab_r = st.columns([1, 2])
    with col_lab_l:
        mech_type = st.radio("실험할 유체 변형 물리 작용", ["오메가 효과 (차동 회전에 의한 수평 감김)", "알파 효과 (코리올리 회오리에 의한 수직 꼬임)"])
        lab_distortion = st.slider("물리적 유체 왜곡 강도 (Distortion)", 0.0, 4.0, 2.0, 0.2)
    with col_lab_r:
        g_res = 30
        lx = np.linspace(-2, 2, g_res)
        ly = np.linspace(-2, 2, g_res)
        LX, LY = np.meshgrid(lx, ly)
        LR = np.sqrt(LX**2 + LY**2)
        
        fig_lab, ax_lab = plt.subplots(figsize=(6, 4.5))
        core_line = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', linewidth=1.5)
        ax_lab.add_patch(core_line)
        
        if "오메가" in mech_type:
            st.info("🌀 **오메가 효과 원리**: 자전 속도 구배(차동 회전)로 인해 원래 일직선이던 자력선 기둥이 수평 동서 방향으로 팽팽하게 실타래처럼 감기며 토로이달 자기장으로 증폭 변형됩니다.")
            Ux = -LY * np.exp(-LR**2) * lab_distortion
            Uy = 1.0 + LX * np.exp(-LR**2) * lab_distortion
            ax_lab.streamplot(LX, LY, Ux, Uy, color='darkorange', density=1.1, linewidth=1.1)
        else:
            st.info("🌪️ **알파 효과 원리**: 회오리 치는 코리올리 열대류 유동이 수평 상태인 자기선을 수직 평면 루프로 꼬아 올리며 다시 남북 종축 방향의 폴로이달 극성을 재생산합니다.")
            twist_cell = np.exp(-(LX**2 + LY**2) / 0.5)
            Ux = 1.0 - LY * twist_cell * lab_distortion
            Uy = LX * twist_cell * lab_distortion
            ax_lab.streamplot(LX, LY, Ux, Uy, color='dodgerblue', density=1.1, linewidth=1.1)
            
        ax_lab.set_xlim(-2, 2)
        ax_lab.set_ylim(-2, 2)
        ax_lab.set_aspect('equal')
        ax_lab.grid(True, linestyle=':', alpha=0.5)
        st.pyplot(fig_lab)
        plt.close(fig_lab)

# 웹 페이지 하단 푸터 영역
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>지구 다이나모 수치 시뮬레이션 | 물리 및 우주지구과학 교육용 웹 애플리케이션</p>", unsafe_allow_html=True)
