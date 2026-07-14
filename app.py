import streamlit as st
import numpy as np
import plotly.graph_objects as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

# 페이지 기본 설정
st.set_page_config(
    page_title="지구 다이나모 이론 시뮬레이터",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사용자 정의 CSS 스타일링 (모바일 대응 및 가독성 향상)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e222b;
        border-radius: 4px;
        color: #ffffff;
        padding-left: 16px;
        padding-right: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b !important;
        font-weight: bold;
    }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("🌍 지구 다이나모 이론 (Geodynamo Theory) 시뮬레이터")
st.subheader("지구 내부의 유체 운동이 어떻게 거대한 우주적 자석을 만드는가?")

# 사이드바 설정 (실시간 변수 조정)
st.sidebar.header("🛠️ 다이나모 시뮬레이션 매개변수")
st.sidebar.markdown("이 변수들이 지구 외핵 내의 발전기(Dynamo) 작용을 결정합니다.")

alpha_strength = st.sidebar.slider(
    "알파(α) 효과 강도 (Helicity)", 
    min_value=0.0, max_value=5.0, value=2.0, step=0.1,
    help="외핵 유체의 대류와 코리올리 효과로 인해 자기력선이 꼬이면서 폴로이달 자기장을 생성하는 효율입니다."
)

omega_strength = st.sidebar.slider(
    "오메가(Ω) 효과 강도 (Differential Rotation)", 
    min_value=0.0, max_value=15.0, value=8.0, step=0.5,
    help="지구 자전 속도의 차이(차등 회전)로 인해 폴로이달 자기장을 끌어당겨 동서 방향의 토로이달 자기장으로 변환하는 힘입니다."
)

magnetic_diffusivity = st.sidebar.slider(
    "자기 확산도 (Magnetic Diffusivity, η)", 
    min_value=0.1, max_value=3.0, value=1.0, step=0.1,
    help="전도성 유체 내에서 자기장이 소멸하고 흩어지는 비율입니다. 값이 너무 크면 다이나모가 유지되지 않고 붕괴합니다."
)

steps = st.sidebar.slider("시뮬레이션 시간 단계 (Steps)", min_value=50, max_value=300, value=150, step=10)

# 메인 탭 구조 설정
tab1, tab2, tab3 = st.tabs(["📖 다이나모 이론 학습", "📊 실시간 다이나모 시뮬레이션", "🌐 3D 지구 자기력선 시각화"])

# -------------------------------------------------------------------------
# TAB 1: 다이나모 이론 학습
# -------------------------------------------------------------------------
with tab1:
    st.header("🧠 다이나모 이론 (Geodynamo Theory) 이란?")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        지구의 자기장은 단순히 자성을 띤 거대한 광물이 안에 들어있기 때문이 아닙니다. 지구 내부 깊은 곳(외핵)은 **섭씨 4,000도가 넘는 액체 금속(철과 니켈)**으로 가득 차 있으며, 이 액체 금속이 끊임없이 흐르며 전류를 만들어내고, 그 전류가 다시 자기장을 만드는 **'자가 발전기(Self-exciting Dynamo)'** 구조를 이루고 있습니다. 이를 **다이나모 이론**이라고 합니다.
        
        다이나모가 지속적으로 작동하기 위해서는 다음과 같은 세 가지 조건이 필요합니다.
        1. **도체인 유체**: 전류를 흘릴 수 있는 외핵의 액체 철.
        2. **에너지원**: 외핵 상하부의 온도 차이로 인해 생기는 **열대류** 및 화학적 성분 차이로 인한 **조성 대류**.
        3. **회전력**: 지구 자전으로 인해 유체 흐름을 비틀어주는 **코리올리 힘(Coriolis force)**.
        """)
    
    with col2:
        # 개념 전달을 위한 플레이스홀더 성격의 시각 프레임 구현
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
        * **역할**: 알파($\alpha$) 효과의 원료가 됩니다.
        """)

    st.markdown("---")
    st.subheader("⚙️ 순환 메커니즘: $\alpha$ 효과와 $\Omega$ 효과")
    
    col_alpha, col_omega = st.columns(2)
    
    with col_omega:
        st.markdown("### 🌀 오메가 효과 ($\Omega$-Effect) : $B_p \\rightarrow B_t$")
        st.markdown("""
        * **원리**: 지구 자전 시 적도 부근과 극 부근의 회전 속도가 다른 **차등 회전(Differential Rotation)**이 발생합니다.
        * **현상**: 이 속도 차이로 인해 남북 방향의 폴로이달 자기선이 동서 방향으로 팽팽하게 감기면서 강한 **토로이달 자기장**으로 변환됩니다.
        """)
        st.latex(r"\frac{\partial B_t}{\partial t} \approx \Delta \Omega \cdot B_p + \eta \nabla^2 B_t")

    with col_alpha:
        st.markdown("### 🌪️ 알파 효과 ($\alpha$-Effect) : $B_t \\rightarrow B_p$")
        st.markdown("""
        * **원리**: 뜨거운 외핵 유체가 상승할 때, 지구 자전에 의한 **코리올리 힘**을 받아 소용돌이치며 회전 상승합니다(나선형 유동, Helicity).
        * **현상**: 이 나선형 흐름이 동서 방향의 토로이달 자기선을 잡아서 고리 모양으로 비틀어 올립니다. 수많은 미세 소용돌이가 만든 고리들이 합쳐져 다시 거대한 **폴로이달 자기장**을 재생산합니다.
        """)
        st.latex(r"\frac{\partial B_p}{\partial t} \approx \alpha \cdot B_t + \eta \nabla^2 B_p")

    st.success("✅ **결론**: 폴로이달 자기장이 $\Omega$ 효과로 토로이달이 되고, 토로이달 자기장이 $\alpha$ 효과로 다시 폴로이달이 되는 이 순환 고리($\alpha\Omega$-Dynamo) 덕분에 지구 자기장은 수십억 년 동안 꺼지지 않고 유지될 수 있습니다.")

# -------------------------------------------------------------------------
# TAB 2: 실시간 다이나모 시뮬레이션
# -------------------------------------------------------------------------
with tab2:
    st.header("📊 $\alpha\Omega$-다이나모 수치 시뮬레이션")
    st.markdown("""
    지구 외핵의 자전축 반경 방향을 단순화한 1차원 공간 격자상에서 평균장 다이나모 방정식(Mean-field Dynamo Equations)을 풀이합니다.
    조정하신 **알파 강도, 오메가 강도, 자기 확산도**에 따라 다이나모가 지속되어 진동하는지, 기하급수적으로 폭발하는지, 혹은 확산되어 사라지는지 실시간으로 관찰해 보세요.
    """)

    # --- 수치 계산 엔진 ---
    # 파라미터 기반 물리 행렬 정의 및 순방향 시간 차분법(Euler-Forward FTCS) 해석
    N = 50  # 공간 격자 크기
    dx = 1.0 / (N - 1)
    dt = 0.001  # 안정성 조건을 고려한 시간 간격
    
    # 초기 상태 정의 (미세한 섭동 자기장)
    Bp = np.sin(np.pi * np.linspace(0, 1, N)) * 0.1
    Bt = np.zeros(N)
    
    # 시간 변화 기록용 배열
    history_time = []
    history_Bp_energy = []
    history_Bt_energy = []
    
    # 수치 안정성 체크 (CFL 조건 간소화 반영)
    stability_factor = magnetic_diffusivity * dt / (dx**2)
    if stability_factor > 0.5:
        st.warning(f"⚠️ 경고: 현재 설정은 자기 확산 조건에 비해 시간 간격이 큽니다 (안정성 인자: {stability_factor:.2f} > 0.5). 수치적 노이즈가 발생할 수 있습니다.")

    # 발전기 시뮬레이션 루프 실행
    for t_step in range(steps):
        new_Bp = np.copy(Bp)
        new_Bt = np.copy(Bt)
        
        # 내부 격자점 업데이트 (경계 조건은 완전 전도체 0 가정)
        for i in range(1, N-1):
            # 2차 도함수 (확산 항)
            d2Bp = (Bp[i+1] - 2*Bp[i] + Bp[i-1]) / (dx**2)
            d2Bt = (Bt[i+1] - 2*Bt[i] + Bt[i-1]) / (dx**2)
            
            # 알파 효과 및 오메가 효과 물리 방정식 대입
            # Bp_dot = alpha * Bt + eta * d2Bp
            # Bt_dot = omega * dBp/dx + eta * d2Bt
            dBp_dx = (Bp[i+1] - Bp[i-1]) / (2*dx)
            
            new_Bp[i] = Bp[i] + dt * (alpha_strength * Bt[i] + magnetic_diffusivity * d2Bp)
            new_Bt[i] = Bt[i] + dt * (omega_strength * dBp_dx + magnetic_diffusivity * d2Bt)
            
        Bp = new_Bp
        Bt = new_Bt
        
        # 총 에너지(L2 Norm 제곱) 기록
        history_time.append(t_step * dt)
        history_Bp_energy.append(np.sum(Bp**2))
        history_Bt_energy.append(np.sum(Bt**2))

    # --- 결과 시각화 ---
    col_fig1, col_fig2 = st.columns(2)
    
    with col_fig1:
        st.subheader("📈 시간 흐름에 따른 자기장 에너지 변화")
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(x=history_time, y=history_Bp_energy, name="폴로이달 에너지 (Bp^2)", line=dict(color='#ff4b4b', width=3)))
        fig_energy.add_trace(go.Scatter(x=history_time, y=history_Bt_energy, name="토로이달 에너지 (Bt^2)", line=dict(color='#00f0ff', width=3)))
        fig_energy.update_layout(
            template="plotly_dark",
            xaxis_title="무차원 시간 (Time)",
            yaxis_title="자기장 에너지 (Energy)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_energy, use_container_width=True)
        
    with col_fig2:
        st.subheader("📐 핵 반경 방향에 따른 최종 자기장 분포")
        # 현재 마지막 스텝의 Bp, Bt 프로파일
        fig_profile = go.Figure()
        r_grid = np.linspace(0.3, 1.0, N) # 외핵 안쪽(0.3)부터 외핵 바깥 경계(1.0)
        fig_profile.add_trace(go.Scatter(x=r_grid, y=Bp, name="폴로이달 성분 (Bp)", line=dict(color='#ff4b4b', dash='dash')))
        fig_profile.add_trace(go.Scatter(x=r_grid, y=Bt, name="토로이달 성분 (Bt)", line=dict(color='#00f0ff')))
        fig_profile.update_layout(
            template="plotly_dark",
            xaxis_title="지구 반경 (외핵 내 규격화 반경 r)",
            yaxis_title="자기장 세기 (Field Strength)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_profile, use_container_width=True)

    # 물리적 상태 해석 피드백 제공
    st.markdown("### 🔍 시뮬레이션 결과 물리 해석")
    final_bp_energy = history_Bp_energy[-1]
    final_bt_energy = history_Bt_energy[-1]
    
    if final_bp_energy < 0.001:
        st.error("📉 **다이나모 소멸 (Decay)**: 자기 확산도($\\eta$)가 생산력($\\alpha \\times \\Omega$)보다 강해 자기장이 유지되지 못하고 소멸했습니다. 지구 자기장이 꺼진 상태입니다!")
    elif final_bp_energy > 10.0:
        st.warning("📈 **폭발적 발산 (Unbounded Growth)**: 실제 지구는 에너지가 무한히 증가할 때 유동 역반응(Lorentz force)이 작용해 포화(Saturation)가 일어나지만, 현재 선형 모델에서는 에너지가 제어할 수 없이 성장하고 있습니다.")
    else:
        st.success("🔄 **지속 가능한 진동형 다이나모 (Self-Sustaining Dynamo)**: $\\alpha$ 효과와 $\\Omega$ 효과가 확산과 절묘한 조화를 이루어 안정한 주기적 상호 변환 상태를 이룹니다! 지구 자기장이 안정적으로 켜져 있습니다.")

# -------------------------------------------------------------------------
# TAB 3: 3D 지구 자기력선 시각화
# -------------------------------------------------------------------------
with tab3:
    st.header("🌐 지구 주변의 3차원 자기력선 뷰")
    st.markdown("""
    외핵에서 생성된 폴로이달 자기장은 행성 외부로 탈출하여 우주로 뻗어나가는 거대한 쌍극자(Dipole) 자기장을 형성합니다.
    아래 3D 플롯은 시뮬레이션에서 생성된 최종 **폴로이달 성분의 강도**와 연계된 지구 주변의 가상 자기력선 분포입니다.
    *(마우스로 회전, 줌인/아웃이 가능하며 모바일 기기에서는 손가락 터치 제스처로 조작할 수 있습니다)*
    """)

    # 3D 자기력선(Dipole Field Line) 수치 생성 함수
    def get_dipole_lines(strength, num_lines=28, num_points=60):
        # 쌍극자 공식: r = R * sin^2(theta)
        lines = []
        # 지구 자기장 세기가 아주 낮을 경우 최소 강도 고정
        effective_strength = max(strength, 0.1) * 2.5
        
        # 일정한 구면 위상 분할
        for i in range(num_lines):
            # 동서 평면 각도 (경도)
            phi = (i / num_lines) * 2 * np.pi
            
            # 각 라인마다 다른 최대 고리 반경을 가짐
            R_max = (0.8 + 1.5 * np.random.rand()) * (1.0 + np.sqrt(effective_strength) * 0.5)
            
            theta = np.linspace(0.05, np.pi - 0.05, num_points)
            r = R_max * (np.sin(theta) ** 2)
            
            # 데카르트 좌표계 변환
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            lines.append((x, y, z))
        return lines

    # 최종 폴로이달 값의 세기 연동
    current_strength = np.max(np.abs(Bp)) if len(Bp) > 0 else 1.0
    field_lines = get_dipole_lines(current_strength)

    # Plotly 3D 그래픽 설정
    fig_3d = go.Figure()

    # 1. 지구 표면 구체(Sphere) 생성
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    # 지구의 반지름 규격 r = 1
    xs = 1.0 * np.outer(np.cos(u), np.sin(v))
    ys = 1.0 * np.outer(np.sin(u), np.sin(v))
    zs = 1.0 * np.outer(np.ones(np.size(u)), np.cos(v))

    # 지구 시각화
    fig_3d.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0, '#0d2b45'], [0.5, '#203c56'], [1, '#546a7b']],
        showscale=False,
        name="지구 (Earth)",
        opacity=0.9,
        hoverinfo='none'
    ))

    # 2. 3D 자기력선 추가
    for idx, (x, y, z) in enumerate(field_lines):
        # 흐름 방향을 보여주기 위해 선에 점차적으로 색상 부여 (남극에서 출발해 북극으로 복귀하는 느낌)
        fig_3d.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(
                color='rgba(135, 206, 250, 0.7)',
                width=3
            ),
            name=f"자기력선 {idx+1}" if idx == 0 else "",
            showlegend=True if idx == 0 else False,
            hoverinfo='none'
        ))

    # 자전축 방향 벡터 표시
    fig_3d.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-2.5, 2.5],
        mode='lines+text',
        line=dict(color='white', width=4, dash='dash'),
        name="자전축 (Rotation Axis)",
        text=["", "지구 자전축(N)"],
        textposition="top center"
    ))

    # Layout 세부 튜닝
    fig_3d.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='X', showbackground=False, visible=False),
            yaxis=dict(title='Y', showbackground=False, visible=False),
            zaxis=dict(title='Z', showbackground=False, visible=False),
            aspectmode='data',
            camera=dict(
                eye=dict(x=2.0, y=2.0, z=1.5) # 비스듬하게 지구를 내려다보는 구도
            )
        ),
        height=600
    )

    col_3d_left, col_3d_right = st.columns([3, 1])
    with col_3d_left:
        st.plotly_chart(fig_3d, use_container_width=True)
    with col_3d_right:
        st.markdown("### 💡 시각화 상세 제어")
        st.markdown(f"**현재 다이나모 유도강도 지수**: `{current_strength:.4f}`")
        st.markdown("""
        * **자기장의 팽창**: 사이드바에서 **알파 강도**나 **오메가 강도**를 키우면 외핵 내부 에너지가 증가하여 외부에 펼쳐지는 자기력선의 규모가 눈에 띄게 우람해집니다.
        * **자기장의 소멸**: 반대로 **자기 확산도($\\eta$)**를 대폭 올린 채로 시뮬레이션을 다시 돌려보세요. 외부로 뻗어나가는 파란색 자기력선의 밀도와 부피가 축소될 것입니다.
        """)
        
        # 사용자를 위한 시뮬레이션 초기화 팁 안내
        st.info("💡 **팁**: 매개변수를 조절한 뒤 시뮬레이션 결과가 즉시 반영되도록 실시간 시뮬레이션 탭과 연계되어 작동합니다.")

# 푸터 영역 정의
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>지구 다이나모 수치 시뮬레이션 | 물리 교육용 웹 애플리케이션</p>", unsafe_allow_html=True)
