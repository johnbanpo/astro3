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
# 6. 신규 추가: 3D 지구 자기력선 및 메커니즘 렌더링 영역 정의
# ==========================================
st.markdown("---")
st.header("🔮 [신규 기능] 3D 입체 자기력선 및 다이나모 개별 메커니즘 시각화")
st.markdown("상단의 수치 연산 프레임과 실시간 연동되거나, 개별 물리 법칙을 독립적으로 학습할 수 있는 공간입니다.")

col3, col4 = st.columns(2)

with col3:
    st.subheader("🌍 3차원 지구 모형과 우주 자기력선 보호막")
    st.caption("실제 나침반 자석 원리에 기반한 대칭형 3D 자기력선 궤적입니다. 연산 강도에 따라 크기가 호흡하듯 변화합니다.")
    graph_3d = st.empty()

with col4:
    st.subheader("🌪️ α 효과 & Ω 효과 독립 메커니즘")
    st.caption("슬라이더를 통해 외핵 내부의 개별 유체 변형 왜곡 상태를 직관적으로 관찰합니다.")
    mech_choice = st.selectbox("관찰할 물리 현상 선택", ["오메가 효과 (Ω-Effect): 차동 회전에 의한 자기선 감김", "알파 효과 (α-Effect): 코리올리 소용돌이에 의한 자기선 꼬임"])
    distortion_val = st.slider("물리적 변형 왜곡도", 0.0, 4.0, 2.0, 0.2)
    graph_mech = st.empty()


# 💡 [핵심 대개편] NASA 및 교육 도식을 참조한 직관적인 3D 자기력선 렌더링 함수
def render_3d_magnetic_field(bp_intensity):
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 3D 지구본 디자인 (반경 0.5로 정밀 제어, 투명도를 주어 입체감 부여)
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x_earth = 0.5 * np.outer(np.cos(u), np.sin(v))
    y_earth = 0.5 * np.outer(np.sin(u), np.sin(v))
    z_earth = 0.5 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, color='dodgerblue', alpha=0.4, edgecolor='w', linewidth=0.1)
    
    # 2. 지구 내부의 자석 축 막대기 표현 (N극: 빨강 / S극: 파랑)
    ax.plot([0, 0], [0, 0], [0.5, 0.9], color='red', linewidth=4, label='Magnetic North')
    ax.plot([0, 0], [0, 0], [-0.5, -0.9], color='blue', linewidth=4, label='Magnetic South')
    
    # 3. 우주 공간으로 이쁘게 늘어지는 다이폴(Dipole) 자기력선 수식 적용
    # r = R0 * sin^2(theta) 공식을 사용하여 완벽한 루프 궤적 계산
    # 수치연산 강도를 스케일에 부드럽게 매핑하되, 선이 지구 안으로 숨지 않도록 최소 크기(0.8) 안전 보장
    raw_scale = np.mean(np.abs(bp_intensity))
    scale = 0.8 + np.clip(raw_scale * 0.8, 0.0, 1.5)
    
    # 12개의 경도 방향으로 일정하게 방사형 배치
    longitudes = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    # 루프 고리들의 크기 단계 설정
    loop_sizes = [0.8, 1.3, 1.9, 2.6]
    
    for lon in longitudes:
        for r_max_base in loop_sizes:
            r_max = r_max_base * scale
            
            # 지구 내부 통과 시점을 피하기 위한 각도 제한 연산
            # sin^2(theta) * r_max > 0.5 (지구본 밖으로 나오는 구간만 그리기)
            min_val = np.clip(0.5 / r_max, 0, 1)
            theta_min = np.arcsin(np.sqrt(min_val))
            
            # 극과 극을 잇는 부드러운 각도 범위 설정
            theta = np.linspace(theta_min, np.pi - theta_min, 40)
            r = r_max * (np.sin(theta) ** 2)
            
            # 구면좌표계를 카테시안(3D 직교좌표)으로 변환
            x_line = r * np.sin(theta) * np.cos(lon)
            y_line = r * np.sin(theta) * np.sin(lon)
            z_line = r * np.cos(theta)
            
            # 입체감 넘치고 시인성이 높은 색상(네온 화이트-하늘색 계열 혹은 붉은색) 적용
            ax.plot(x_line, y_line, z_line, color='deepskyblue', alpha=0.5, linewidth=1.1)
            
            # 화살표(흐름 방향) 대용으로 중심 적도 지점에 작고 선명한 인디케이터 점 표현
            mid = len(theta) // 2
            ax.scatter(x_line[mid], y_line[mid], z_line[mid], color='cyan', s=6, alpha=0.8)
            
    # 시각화 박스 경계 설정
    ax.set_xlim(-3
