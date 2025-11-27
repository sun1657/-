import streamlit as st              # 웹 애플리케이션 프레임워크
import pandas as pd                 # 데이터 처리
import numpy as np                  # 수치 계산
import matplotlib.pyplot as plt     # 기본 차트
import seaborn as sns              # 고급 시각화
import plotly.express as px        # 인터랙티브 차트
import plotly.graph_objects as go  # 커스텀 plotly 차트
from plotly.subplots import make_subplots  # 서브플롯 생성
import os
import warnings

# 경고 메시지 숨김
warnings.filterwarnings('ignore')

MODULES_LOADED = False  # 모듈 로딩 상태 플래그
try:
    from physionet_predictor import *    # 혈압 예측 모델
    from langchain_processor import *    # AI 분석 엔진
    MODULES_LOADED = True
except ImportError as e:
    # 모듈이 없어도 기본 기능은 작동하도록 설계
    st.warning(f"⚠️ 모듈 임포트 경고: {e}")

# =========================================
# Streamlit 페이지 설정
# =========================================
st.set_page_config(
    page_title="혈압 예측 AI 시스템",  # 브라우저 탭 제목
    page_icon="🩺",                    # 파비콘
    layout="wide",                     # 넓은 레이아웃
    initial_sidebar_state="expanded"   # 사이드바 기본 열림
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# =========================================
# CSS 스타일링
# =========================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# 메인 헤더 & 사이드바
# =========================================
# 메인 헤더
st.markdown('<h1 class="main-header">🩺 혈압 예측 AI 시스템</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 2rem;">
PhysioNet 실제 의료 데이터 기반 머신러닝 모델과 LangChain AI 분석을 통한 개인 맞춤형 혈압 관리 솔루션
</div>
""", unsafe_allow_html=True)

# 사이드바 - 모델 초기화 및 설정
st.sidebar.markdown("## ⚙️ 시스템 설정")

# =========================================
# 모델 로드 & 데이터 로드
# =========================================
# 모델 로드 상태 확인
@st.cache_resource
def load_models():
    """모델들을 로드하고 캐시"""
    if not MODULES_LOADED:
        return None, None
    
    try:
        predictor = PhysioNetBPPredictor()
        processor = LangChainBPProcessor()
        return predictor, processor
    except Exception as e:
        st.sidebar.error(f"모델 로드 실패: {e}")
        return None, None

@st.cache_data
def load_data():
    """데이터를 로드하고 캐시"""
    data_paths = [
        'all_patient_features_preprocessed.csv',
        'all_patient_features.csv',
        'data/processed_bp_data.csv'
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ 데이터 로드: {path}")
                return df, True
            except Exception as e:
                st.sidebar.warning(f"⚠️ {path} 로드 실패: {e}")
                continue
    
    st.sidebar.warning("⚠️ 데이터 파일을 찾을 수 없습니다.")
    return None, False

predictor, processor = load_models()
sample_data, data_loaded = load_data()

# =========================================
# 모델 상태 표시
# =========================================
if MODULES_LOADED and predictor and processor:
    st.sidebar.success("✅ 모델 로드 완료")
    # LLM 사용 가능 여부 표시
    if processor.llm is not None:
        st.sidebar.success("✅ AI 분석 엔진 활성화 (JSON mode)")
    else:
        st.sidebar.info("ℹ️ AI 분석: 기본 알고리즘 사용")
        st.sidebar.caption("💡 OpenAI API 키를 설정하면 고급 AI 분석을 사용할 수 있습니다.")
elif MODULES_LOADED:
    st.sidebar.warning("⚠️ 모델 초기화 중 일부 문제 발생")
else:
    st.sidebar.info("ℹ️ 기본 분석 모드로 실행")

# =========================================
# 탭 생성 - 4개
# =========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 홈", 
    "🔮 혈압 예측", 
    "📊 데이터 분석", 
    "🤖 AI 건강 상담"
])

# ============================================================
# 탭 1: 홈
# ============================================================
with tab1:
    st.markdown('<h2 class="section-header">🏠 시스템 개요</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 정확한 예측</h4>
            <p>PhysioNet 실제 의료 데이터로 훈련된 머신러닝 모델</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 AI 분석</h4>
            <p>LangChain 기반 개인 맞춤형 건강 조언</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 시각화</h4>
            <p>직관적인 차트와 그래프로 결과 확인</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h3 class="section-header">🚀 시작하기</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 사용 방법:
    1. **혈압 예측** 탭에서 환자 정보를 입력하세요
    2. **데이터 분석** 탭에서 전체 데이터셋을 탐색하세요  
    3. **AI 건강 상담** 탭에서 개인 맞춤형 조언을 받으세요
    
    ### ⚠️ 주의사항:
    - 이 시스템은 교육 및 참고 목적으로만 사용됩니다
    - 실제 의료 진단이나 치료를 대체할 수 없습니다
    - 건강 문제가 있으시면 반드시 의료진과 상담하세요
    """)
    
    # AI 기능 상태 안내
    if processor and processor.llm:
        st.markdown("""
        <div class="success-box">
            <h4>✅ AI 분석 엔진 활성화</h4>
            <p>GPT-4o-mini를 사용한 고급 AI 분석이 가능합니다.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>ℹ️ 기본 분석 모드</h4>
            <p>현재 기본 알고리즘을 사용하고 있습니다. OpenAI API 키를 설정하면 더 정교한 AI 분석을 받을 수 있습니다.</p>
            <p><code>.env</code> 파일에 <code>OPENAI_API_KEY=your-key-here</code>를 추가하세요.</p>
        </div>
        """, unsafe_allow_html=True)
    
    if data_loaded and sample_data is not None:
        st.markdown('<h3 class="section-header">📈 데이터 현황</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 환자 수", f"{len(sample_data):,}명")
        
        with col2:
            age_cols = ['age', 'Age', 'AGE']
            age_col = next((col for col in age_cols if col in sample_data.columns), None)
            if age_col:
                avg_age = sample_data[age_col].mean()
                st.metric("평균 연령", f"{avg_age:.1f}세")
            else:
                st.metric("데이터 컬럼 수", f"{len(sample_data.columns)}개")
        
        with col3:
            bp_cols = ['systolic_bp', 'NIBP_mean', 'NIBP_max', 'bp_systolic']
            bp_col = next((col for col in bp_cols if col in sample_data.columns), None)
            if bp_col:
                avg_bp = sample_data[bp_col].mean()
                st.metric("평균 혈압", f"{avg_bp:.1f} mmHg")
            else:
                st.metric("데이터 특성", "PhysioNet")
        
        with col4:
            bmi_cols = ['bmi', 'BMI']
            bmi_col = next((col for col in bmi_cols if col in sample_data.columns), None)
            if bmi_col:
                avg_bmi = sample_data[bmi_col].mean()
                st.metric("평균 BMI", f"{avg_bmi:.1f}")
            else:
                if 'sampling_rate' in sample_data.columns:
                    avg_sr = sample_data['sampling_rate'].mean()
                    st.metric("샘플링 레이트", f"{avg_sr:.0f} Hz")
                else:
                    st.metric("상태", "정상")
                    
# ============================================================
# 탭 2: 혈압 예측
# ============================================================
with tab2:
    st.markdown('<h2 class="section-header">🔮 개인 혈압 예측</h2>', unsafe_allow_html=True)
    
    with st.form("patient_form"):
        st.markdown("### 📝 환자 정보 입력")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("나이 (세)", min_value=18, max_value=100, value=45)
            gender = st.selectbox("성별", ["남성", "여성"])
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=23.0, step=0.1)
            smoking = st.selectbox("흡연 여부", ["비흡연", "흡연"])
        
        with col2:
            exercise_frequency = st.number_input("주간 운동 횟수", min_value=0, max_value=7, value=2)
            stress_level = st.slider("스트레스 수준 (1-10)", min_value=1, max_value=10, value=5)
            heart_rate = st.number_input("심박수 (bpm)", min_value=50, max_value=120, value=75)
            family_history = st.multiselect("가족력", ["고혈압", "당뇨병", "심장병", "없음"])
        
        st.markdown("### 🩸 현재 혈압 (선택사항)")
        col1, col2 = st.columns(2)
        with col1:
            current_systolic = st.number_input("수축기 혈압 (mmHg)", min_value=80, max_value=200, value=120)
        with col2:
            current_diastolic = st.number_input("이완기 혈압 (mmHg)", min_value=50, max_value=120, value=80)
        
        submitted = st.form_submit_button("🔮 혈압 예측하기", )
    
    if submitted:
        patient_data = {
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'smoking': 1 if smoking == "흡연" else 0,
            'exercise_frequency': exercise_frequency,
            'stress_level': stress_level,
            'heart_rate_bpm': heart_rate,
            'family_history_hypertension': 1 if "고혈압" in family_history else 0,
            'family_history_diabetes': 1 if "당뇨병" in family_history else 0,
            'systolic_bp': current_systolic,
            'diastolic_bp': current_diastolic
        }
        width='stretch'
        with st.spinner("🔄 혈압 예측 중..."):
            try:
                if predictor and MODULES_LOADED:
                    prediction_result = predictor.predict(patient_data)
                else:
                    # 기본 예측 알고리즘
                    base_systolic = 100 + (age * 0.5)
                    base_diastolic = 60 + (age * 0.3)
                    
                    if bmi >= 30:
                        base_systolic += 10
                        base_diastolic += 5
                    elif bmi >= 25:
                        base_systolic += 5
                        base_diastolic += 3
                    
                    if smoking == "흡연":
                        base_systolic += 5
                        base_diastolic += 3
                    
                    if exercise_frequency < 2:
                        base_systolic += 3
                        base_diastolic += 2
                    
                    if stress_level >= 7:
                        base_systolic += 5
                        base_diastolic += 3
                    
                    prediction_result = {
                        'systolic_bp': base_systolic,
                        'diastolic_bp': base_diastolic,
                        'model_used': '기본 알고리즘'
                    }
                
                st.markdown('<h3 class="section-header">📊 예측 결과</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "예측 수축기 혈압", 
                        f"{prediction_result['systolic_bp']:.1f} mmHg",
                        delta=f"{prediction_result['systolic_bp'] - current_systolic:.1f}" if current_systolic != 120 else None
                    )
                
                with col2:
                    st.metric(
                        "예측 이완기 혈압", 
                        f"{prediction_result['diastolic_bp']:.1f} mmHg",
                        delta=f"{prediction_result['diastolic_bp'] - current_diastolic:.1f}" if current_diastolic != 80 else None
                    )
                
                with col3:
                    st.metric("사용 모델", prediction_result['model_used'])
                
                # 혈압 분류
                systolic = prediction_result['systolic_bp']
                diastolic = prediction_result['diastolic_bp']
                
                if systolic >= 140 or diastolic >= 90:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>🚨 고혈압 (1기 이상)</h4>
                        <p>의료진 상담을 받으시기 바랍니다.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif systolic >= 120 or diastolic >= 80:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ 고혈압 전단계</h4>
                        <p>생활습관 개선이 필요합니다.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <h4>✅ 정상 혈압</h4>
                        <p>현재 상태를 유지하세요.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 시각화
                st.markdown('<h4 class="section-header">📈 혈압 분포 시각화</h4>', unsafe_allow_html=True)
                
                fig = go.Figure()
                
                # 정상 범위
                fig.add_shape(
                    type="rect",
                    x0=90, y0=60, x1=120, y1=80,
                    fillcolor="green",
                    opacity=0.2,
                    line=dict(color="green", width=2),
                )
                
                # 고혈압 전단계
                fig.add_shape(
                    type="rect",
                    x0=120, y0=60, x1=140, y1=90,
                    fillcolor="yellow",
                    opacity=0.2,
                    line=dict(color="orange", width=2),
                )
                
                # 고혈압
                fig.add_shape(
                    type="rect",
                    x0=140, y0=60, x1=200, y1=120,
                    fillcolor="red",
                    opacity=0.2,
                    line=dict(color="red", width=2),
                )
                
                # 예측값 표시
                fig.add_trace(go.Scatter(
                    x=[systolic],
                    y=[diastolic],
                    mode='markers',
                    marker=dict(size=15, color='blue', symbol='star'),
                    name='예측값',
                    hovertemplate='<b>예측 혈압</b><br>수축기: %{x:.1f} mmHg<br>이완기: %{y:.1f} mmHg<extra></extra>'
                ))
                
                # 현재값 표시
                if current_systolic != 120 or current_diastolic != 80:
                    fig.add_trace(go.Scatter(
                        x=[current_systolic],
                        y=[current_diastolic],
                        mode='markers',
                        marker=dict(size=15, color='green', symbol='circle'),
                        name='현재값',
                        hovertemplate='<b>현재 혈압</b><br>수축기: %{x:.1f} mmHg<br>이완기: %{y:.1f} mmHg<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="혈압 분포도",
                    xaxis_title="수축기 혈압 (mmHg)",
                    yaxis_title="이완기 혈압 (mmHg)",
                    width=600,
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch', config={'displaylogo': False})
                
            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")
                st.info("💡 기본 분석 알고리즘을 사용하여 재시도합니다.")

# ============================================================
# 탭 3: 데이터 분석
# ============================================================
with tab3:
    st.markdown('<h2 class="section-header">📊 데이터 분석</h2>', unsafe_allow_html=True)
    
    if not data_loaded or sample_data is None:
        st.warning("⚠️ 데이터가 로드되지 않았습니다.")
        st.info("💡 PhysioNet 데이터를 처리하려면 `physionet_predictor.py`를 실행하세요.")
        
        if st.button("📊 샘플 데이터로 시연하기"):
            np.random.seed(42)
            n_samples = 100
            sample_data = pd.DataFrame({
                'age': np.random.randint(20, 80, n_samples),
                'systolic_bp': np.random.randint(100, 160, n_samples),
                'diastolic_bp': np.random.randint(60, 100, n_samples),
                'bmi': np.random.uniform(18, 35, n_samples),
                'heart_rate': np.random.randint(60, 100, n_samples)
            })
            st.session_state['sample_data'] = sample_data
            st.success("✅ 샘플 데이터가 생성되었습니다!")
            st.rerun()
    else:
        st.markdown('<h3 class="section-header">📈 데이터 개요</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 기본 통계")
            numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(sample_data[numeric_cols].describe(), width='stretch')
            else:
                st.info("숫자형 데이터가 없습니다.")
        
        with col2:
            st.markdown("### 데이터 정보")
            st.write(f"**총 레코드 수:** {len(sample_data):,}개")
            st.write(f"**특성 수:** {len(sample_data.columns)}개")
            st.write(f"**결측값:** {sample_data.isnull().sum().sum()}개")
            
            with st.expander("📋 컬럼 목록 보기"):
                for col in sample_data.columns:
                    st.write(f"- {col}")
        
        # 시각화
        st.markdown('<h3 class="section-header">📊 데이터 시각화</h3>', unsafe_allow_html=True)
        
        viz_options = []
        if any(col in sample_data.columns for col in ['age', 'Age']):
            viz_options.append("연령 분포")
        if any(col in sample_data.columns for col in ['gender', 'Gender']):
            viz_options.append("성별 분포")
        if any(col in sample_data.columns for col in ['bmi', 'BMI']):
            viz_options.append("BMI 분포")
        if any(col in sample_data.columns for col in ['systolic_bp', 'NIBP_mean', 'NIBP_max']):
            viz_options.append("혈압 분포")
        
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            viz_options.append("상관관계 분석")
        
        if len(viz_options) == 0:
            st.info("시각화 가능한 데이터가 없습니다.")
        else:
            viz_option = st.selectbox("시각화 유형 선택", viz_options)
            
            try:
                if viz_option == "연령 분포":
                    age_col = next((col for col in ['age', 'Age'] if col in sample_data.columns), None)
                    if age_col:
                        fig = px.histogram(sample_data, x=age_col, nbins=20, title="연령 분포")
                        st.plotly_chart(fig, width='stretch', config={'displaylogo': False})
                
                elif viz_option == "성별 분포":
                    gender_col = next((col for col in ['gender', 'Gender'] if col in sample_data.columns), None)
                    if gender_col:
                        gender_counts = sample_data[gender_col].value_counts()
                        fig = px.pie(values=gender_counts.values, names=gender_counts.index, title="성별 분포")
                        st.plotly_chart(fig, width='stretch', config={'displaylogo': False})
                
                elif viz_option == "BMI 분포":
                    bmi_col = next((col for col in ['bmi', 'BMI'] if col in sample_data.columns), None)
                    if bmi_col:
                        fig = px.histogram(sample_data, x=bmi_col, nbins=20, title="BMI 분포")
                        st.plotly_chart(fig, width='stretch', config={'displaylogo': False})
                
                elif viz_option == "혈압 분포":
                    bp_cols = ['systolic_bp', 'NIBP_mean', 'NIBP_max', 'NIBP_min']
                    available_bp_cols = [col for col in bp_cols if col in sample_data.columns]
                    
                    if len(available_bp_cols) >= 2:
                        fig = make_subplots(rows=1, cols=2, subplot_titles=(available_bp_cols[0], available_bp_cols[1]))
                        fig.add_trace(go.Histogram(x=sample_data[available_bp_cols[0]], name=available_bp_cols[0]), row=1, col=1)
                        fig.add_trace(go.Histogram(x=sample_data[available_bp_cols[1]], name=available_bp_cols[1]), row=1, col=2)
                        fig.update_layout(title="혈압 분포", showlegend=False)
                        st.plotly_chart(fig, width='stretch', config={'displaylogo': False})
                    elif len(available_bp_cols) == 1:
                        fig = px.histogram(sample_data, x=available_bp_cols[0], nbins=30, title=f"{available_bp_cols[0]} 분포")
                        st.plotly_chart(fig, width='stretch', config={'displaylogo': False})
                
                elif viz_option == "상관관계 분석":
                    if len(numeric_cols) > 15:
                        st.info(f"총 {len(numeric_cols)}개 특성 중 주요 15개만 표시합니다.")
                        variances = sample_data[numeric_cols].var().sort_values(ascending=False)
                        selected_cols = variances.head(15).index
                    else:
                        selected_cols = numeric_cols
                    
                    corr_matrix = sample_data[selected_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", 
                                   title="상관관계 매트릭스",
                                   color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, width='stretch', config={'displaylogo': False})
            
            except Exception as e:
                st.error(f"시각화 중 오류 발생: {e}")
                st.info("다른 시각화 옵션을 시도해보세요.")

# ============================================================
# 탭 4: AI 건강 상담
# ============================================================
with tab4:
    st.markdown('<h2 class="section-header">🤖 AI 건강 상담</h2>', unsafe_allow_html=True)
    
    # AI 상태 확인
    if processor and processor.llm:
        st.markdown("""
        <div class="success-box">
            <h4>✅ AI 분석 엔진 활성화</h4>
            <p>GPT-4o-mini를 사용한 개인 맞춤형 건강 분석을 제공합니다.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>ℹ️ 기본 분석 모드</h4>
            <p>현재 기본 알고리즘을 사용하고 있습니다. 더 정교한 AI 분석을 원하시면 OpenAI API 키를 설정하세요.</p>
            <p><strong>설정 방법:</strong> 프로젝트 루트 디렉토리에 <code>.env</code> 파일을 생성하고 <code>OPENAI_API_KEY=your-key-here</code>를 추가하세요.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📝 간단한 환자 정보 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ai_age = st.number_input("나이 (세)", min_value=18, max_value=100, value=45, key="ai_age")
        ai_gender = st.selectbox("성별", ["남성", "여성"], key="ai_gender")
        ai_bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=23.0, step=0.1, key="ai_bmi")
    
    with col2:
        ai_smoking = st.selectbox("흡연 여부", ["비흡연", "흡연"], key="ai_smoking")
        ai_exercise = st.number_input("주간 운동 횟수", min_value=0, max_value=7, value=2, key="ai_exercise")
        ai_stress = st.slider("스트레스 수준 (1-10)", min_value=1, max_value=10, value=5, key="ai_stress")
    
    # 현재 혈압
    st.markdown("### 🩸 현재 혈압")
    col1, col2 = st.columns(2)
    with col1:
        ai_systolic = st.number_input("현재 수축기 혈압 (mmHg)", min_value=80, max_value=200, value=120, key="ai_systolic")
    with col2:
        ai_diastolic = st.number_input("현재 이완기 혈압 (mmHg)", min_value=50, max_value=120, value=80, key="ai_diastolic")
    
    # 분석 유형 선택
    analysis_type = st.radio(
        "분석 유형 선택",
        ["개별 환자 분석", "개인 맞춤 건강 조언"],
        horizontal=True
    )
    
    if st.button("🤖 AI 건강 분석 받기", width='stretch'):
        # 환자 데이터 준비
        patient_data = {
            'age': ai_age,
            'gender': ai_gender,
            'bmi': ai_bmi,
            'smoking': 1 if ai_smoking == "흡연" else 0,
            'exercise_frequency': ai_exercise,
            'stress_level': ai_stress,
            'systolic_bp': ai_systolic,
            'diastolic_bp': ai_diastolic,
            'heart_rate_bpm': 75,  # 기본값
            'family_history_hypertension': 0,
            'family_history_diabetes': 0
        }
        
        # 예측 결과 생성
        prediction_result = {
            'systolic_bp': ai_systolic,
            'diastolic_bp': ai_diastolic
        }
        
        with st.spinner("🤖 AI가 분석 중입니다..."):
            try:
                if analysis_type == "개별 환자 분석":
                    # AI 개별 분석
                    if processor and MODULES_LOADED:
                        analysis = processor.analyze_individual_bp(patient_data)
                    else:
                        # 기본 분석
                        systolic = patient_data['systolic_bp']
                        diastolic = patient_data['diastolic_bp']
                        
                        risk_factors = []
                        recommendations = []
                        
                        if systolic >= 140 or diastolic >= 90:
                            risk_level = "높음"
                            risk_factors.append("고혈압 수치")
                            recommendations.append("즉시 의료진 상담을 받으세요")
                        elif systolic >= 120 or diastolic >= 80:
                            risk_level = "보통"
                            risk_factors.append("고혈압 전단계")
                            recommendations.append("생활습관 개선이 필요합니다")
                        else:
                            risk_level = "낮음"
                            recommendations.append("현재 상태를 유지하세요")
                        
                        if ai_bmi >= 30:
                            risk_factors.append("비만")
                            recommendations.append("체중 감량이 필요합니다")
                        elif ai_bmi >= 25:
                            risk_factors.append("과체중")
                            recommendations.append("적정 체중 유지를 권장합니다")
                        
                        if ai_smoking == "흡연":
                            risk_factors.append("흡연")
                            recommendations.append("금연을 강력히 권장합니다")
                        
                        if ai_exercise < 3:
                            risk_factors.append("운동 부족")
                            recommendations.append("주 3회 이상 규칙적인 운동을 하세요")
                        
                        if ai_stress >= 7:
                            risk_factors.append("높은 스트레스")
                            recommendations.append("스트레스 관리 기법을 실천하세요")
                        
                        if not risk_factors:
                            risk_factors.append("특별한 위험 요인 없음")
                        
                        analysis = {
                            'analysis_type': '기본_분석',
                            'overall_assessment': f"현재 혈압 {systolic}/{diastolic} mmHg, 위험도는 {risk_level}입니다.",
                            'risk_level': risk_level,
                            'key_risk_factors': risk_factors,
                            'recommendations': recommendations,
                            'lifestyle_advice': "규칙적인 운동, 건강한 식단, 스트레스 관리가 중요합니다.",
                            'follow_up_needed': risk_level in ["높음", "매우높음"],
                            'source': '기본_알고리즘'
                        }
                    
                    # 결과 표시
                    st.markdown('<h3 class="section-header">🧠 AI 분석 결과</h3>', unsafe_allow_html=True)
                    
                    # 전반적 평가
                    st.markdown(f"**전반적 평가:** {analysis['overall_assessment']}")
                    
                    # 위험도 표시
                    risk_colors = {
                        "낮음": "🟢",
                        "보통": "🟡", 
                        "높음": "🟠",
                        "매우높음": "🔴"
                    }
                    risk_emoji = risk_colors.get(analysis['risk_level'], "❓")
                    st.markdown(f"**위험도:** {risk_emoji} {analysis['risk_level']}")
                    
                    # 주요 위험 요인
                    st.markdown("### ⚠️ 주요 위험 요인")
                    for factor in analysis['key_risk_factors']:
                        st.markdown(f"• {factor}")
                    
                    # 권장사항
                    st.markdown("### ✅ 권장사항")
                    for i, rec in enumerate(analysis['recommendations'], 1):
                        st.markdown(f"{i}. {rec}")
                    
                    # 생활습관 조언
                    st.markdown("### 💡 생활습관 조언")
                    st.markdown(analysis['lifestyle_advice'])
                    
                    # 추가 검진 필요 여부
                    if analysis['follow_up_needed']:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>🏥 추가 검진 권장</h4>
                            <p>의료진과의 상담을 받으시기 바랍니다.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 분석 출처
                    st.caption(f"분석 출처: {analysis['source']}")
                
                else:  # 개인 맞춤 건강 조언
                    st.markdown('<h3 class="section-header">💡 개인 맞춤 건강 조언</h3>', unsafe_allow_html=True)
                    
                    if processor and MODULES_LOADED:
                        health_advice = processor.generate_health_advice(patient_data, prediction_result)
                        st.markdown(health_advice)
                    else:
                        # 기본 건강 조언
                        st.markdown(f"""
                        **{ai_age}세 {ai_gender} 환자님을 위한 건강 조언:**
                        
                        ### 1. 혈압 관리
                        - 현재 혈압: {ai_systolic}/{ai_diastolic} mmHg
                        - 정기적인 혈압 측정을 권장합니다
                        
                        ### 2. 생활습관 개선
                        - 규칙적인 운동 (주 {ai_exercise}회 → 주 3-5회로 증가)
                        - 건강한 식단 (저염식, 채소와 과일 충분히)
                        - 적정 체중 유지 (현재 BMI: {ai_bmi:.1f})
                        
                        ### 3. 스트레스 관리
                        - 현재 스트레스 수준: {ai_stress}/10
                        - 명상, 요가, 취미활동 등을 통한 스트레스 해소
                        
                        ### 4. 정기 검진
                        - 6개월마다 건강검진 권장
                        - 혈압이 높은 경우 의료진 상담 필수
                        
                        ---
                        
                        ⚠️ 이 조언은 일반적인 건강 가이드이며, 개인 맞춤 상담은 의료진과 하세요.
                        """)
                    
                    # 추가 정보
                    st.markdown("### 📚 추가 정보")
                    
                    with st.expander("혈압 수치 이해하기"):
                        st.markdown("""
                        **정상 혈압:** 수축기 < 120 mmHg, 이완기 < 80 mmHg
                        
                        **고혈압 전단계:** 수축기 120-139 mmHg 또는 이완기 80-89 mmHg
                        
                        **고혈압 1기:** 수축기 140-159 mmHg 또는 이완기 90-99 mmHg
                        
                        **고혈압 2기:** 수축기 ≥ 160 mmHg 또는 이완기 ≥ 100 mmHg
                        """)
                    
                    with st.expander("생활습관 개선 팁"):
                        st.markdown("""
                        **식단:**
                        - 나트륨 섭취 줄이기 (하루 2,300mg 미만)
                        - DASH 식단 (과일, 채소, 저지방 유제품 위주)
                        - 가공식품 피하기
                        
                        **운동:**
                        - 주 150분 이상 중강도 유산소 운동
                        - 걷기, 수영, 자전거 타기 등
                        - 근력 운동 주 2회 이상
                        
                        **기타:**
                        - 금연
                        - 절주 (하루 1-2잔 이하)
                        - 충분한 수면 (7-8시간)
                        - 스트레스 관리
                        """)
                
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")
                st.info("💡 OpenAI API 키를 설정하면 더 정확한 AI 분석을 받을 수 있습니다.")
                
                with st.expander("🔧 API 키 설정 방법"):
                    st.code("""
# 1. 프로젝트 루트에 .env 파일 생성
# 2. 다음 내용 추가:
OPENAI_API_KEY=your-openai-api-key-here

# 3. Streamlit 앱 재시작
                    """, language="bash")
    
    # 데이터셋 AI 인사이트 섹션 (선택사항)
    if data_loaded and sample_data is not None and processor and MODULES_LOADED:
        st.markdown("---")
        st.markdown('<h3 class="section-header">📊 데이터셋 AI 인사이트</h3>', unsafe_allow_html=True)
        
        if st.button("🔍 전체 데이터셋 AI 분석 받기"):
            with st.spinner("🤖 전체 데이터셋을 분석 중입니다..."):
                try:
                    # 샘플링 (너무 큰 경우)
                    if len(sample_data) > 100:
                        analysis_df = sample_data.sample(n=100, random_state=42)
                        st.info(f"전체 {len(sample_data)}명 중 100명을 샘플링하여 분석합니다.")
                    else:
                        analysis_df = sample_data
                    
                    dataset_analysis = processor.analyze_dataset_insights(analysis_df)
                    
                    st.markdown('<h4 class="section-header">📈 분석 결과</h4>', unsafe_allow_html=True)
                    
                    st.markdown(f"**분석 타입:** {dataset_analysis['analysis_type']}")
                    st.markdown(f"**분석된 환자 수:** {dataset_analysis['total_patients']}명")
                    
                    st.markdown("### 📝 전체 요약")
                    st.markdown(dataset_analysis['summary'])
                    
                    if dataset_analysis.get('key_patterns'):
                        st.markdown("### 🔍 주요 패턴")
                        for i, pattern in enumerate(dataset_analysis['key_patterns'], 1):
                            st.markdown(f"{i}. {pattern}")
                    
                    if dataset_analysis.get('statistical_highlights'):
                        st.markdown("### 📊 통계적 주요점")
                        for i, highlight in enumerate(dataset_analysis['statistical_highlights'], 1):
                            st.markdown(f"{i}. {highlight}")
                    
                    if dataset_analysis.get('clinical_implications'):
                        st.markdown("### 🏥 임상적 의미")
                        for i, implication in enumerate(dataset_analysis['clinical_implications'], 1):
                            st.markdown(f"{i}. {implication}")
                    
                    st.caption(f"분석 출처: {dataset_analysis['source']}")
                    
                except Exception as e:
                    st.error(f"데이터셋 분석 중 오류 발생: {e}")