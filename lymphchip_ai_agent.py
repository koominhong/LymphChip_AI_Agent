import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'  # Mac
# matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="미세생리유체칩 약물 동태 분석", page_icon="🧬", layout="wide")

# 세션 상태 초기화
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'xgb_models' not in st.session_state:
    st.session_state.xgb_models = {}
if 'scaler_X' not in st.session_state:
    st.session_state.scaler_X = None
if 'scaler_y' not in st.session_state:
    st.session_state.scaler_y = {}
if 'default_values' not in st.session_state:
    st.session_state.default_values = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = ['Lp_ve', 'K', 'P_oncotic', 'sigma_ve', 'D_gel']
if 'target_names' not in st.session_state:
    # 🔥 올바른 순서: Decay가 ECM보다 먼저!
    st.session_state.target_names = ['Total mass', 'Lymph', 'Blood', 'Decay', 'ECM']
if 'hyperparams' not in st.session_state:
    st.session_state.hyperparams = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1
    }
if 'smoothing_params' not in st.session_state:
    st.session_state.smoothing_params = {
        'enabled': True,
        'method': 'savgol',
        'window_length': 11,
        'poly_order': 3,
        'spline_kind': 'cubic'
    }

# 제외할 시트 목록
EXCLUDE_SHEETS = ['Summary', 'step size', 'Sheet9']

def extract_input_variables_type1(df):
    """Type 1 파일: AF열(31), AH열(33)의 3~7행"""
    try:
        if df.shape[1] < 34:
            return None
        
        input_vars = {}
        for row_idx in range(2, 7):
            var_name = str(df.iloc[row_idx, 31]).strip()
            var_value_str = str(df.iloc[row_idx, 33])
            
            try:
                var_value_clean = var_value_str.split()[0]
                var_value = float(var_value_clean)
                input_vars[var_name] = var_value
                st.success(f"  ✅ {var_name} = {var_value:.2e}")
            except:
                st.warning(f"  ⚠️ {var_name}: '{var_value_str}' 변환 실패")
        
        return input_vars if input_vars else None
    except Exception as e:
        st.error(f"  ❌ 입력 변수 추출 오류: {str(e)}")
        return None

def extract_target_variables_type1(df):
    """Type 1 파일: O~T열(14~19), Time ≤ 72
    올바른 순서: Time, Total mass, Lymph, Blood, Decay, ECM
    """
    try:
        if df.shape[1] < 20:
            return None
        
        # 1행부터 읽기 (0행은 헤더)
        target_df = df.iloc[1:, 14:20].copy()
        
        # 🔥 올바른 순서로 컬럼명 지정!
        target_df.columns = ['Time(h)', 'Total mass', 'Lymph', 'Blood', 'Decay', 'ECM']
        
        for col in target_df.columns:
            target_df[col] = pd.to_numeric(target_df[col], errors='coerce')
        
        target_df = target_df.dropna()
        
        if 'Time(h)' in target_df.columns:
            original_len = len(target_df)
            target_df = target_df[target_df['Time(h)'] <= 72].reset_index(drop=True)
            st.info(f"  🔍 Time 필터링: {original_len}개 → {len(target_df)}개")
        
        return target_df if len(target_df) > 0 else None
    except Exception as e:
        st.error(f"  ❌ 타겟 변수 추출 오류: {str(e)}")
        return None

def detect_file_type(df):
    """파일 타입 자동 감지"""
    if df.shape[1] >= 34:
        if any('lp_ve' in str(df.iloc[i, 31]).lower() for i in range(min(10, df.shape[0]))):
            return "TYPE1"
    return "TYPE2"

def load_and_process_files(uploaded_files):
    """파일 처리 - Time을 입력 변수로 포함"""
    all_X = []
    all_y = []
    total_sheets_processed = 0
    
    for file_idx, uploaded_file in enumerate(uploaded_files):
        try:
            st.markdown(f"## 📄 파일 {file_idx + 1}: {uploaded_file.name}")
            
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=None)
                sheets_to_process = [('default', df)]
            else:
                xl = pd.ExcelFile(uploaded_file)
                st.info(f"📑 총 {len(xl.sheet_names)}개 시트 발견")
                
                valid_sheets = [s for s in xl.sheet_names if s not in EXCLUDE_SHEETS]
                excluded_count = len(xl.sheet_names) - len(valid_sheets)
                
                if excluded_count > 0:
                    excluded = [s for s in xl.sheet_names if s in EXCLUDE_SHEETS]
                    st.warning(f"⏭️ {excluded_count}개 시트 제외: {', '.join(excluded)}")
                
                st.success(f"✅ {len(valid_sheets)}개 시트 처리 예정")
                
                sheets_to_process = [(sheet_name, pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)) 
                                     for sheet_name in valid_sheets]
            
            for sheet_name, df in sheets_to_process:
                with st.expander(f"📊 시트: {sheet_name}", expanded=False):
                    st.info(f"📐 크기: {df.shape[0]}행 × {df.shape[1]}열")
                    
                    file_type = detect_file_type(df)
                    st.info(f"🔍 타입: {file_type}")
                    
                    if file_type == "TYPE1":
                        input_vars = extract_input_variables_type1(df)
                        if not input_vars:
                            st.error(f"  ⚠️ 입력 변수 추출 실패 - 건너뜀")
                            continue
                        
                        target_df = extract_target_variables_type1(df)
                        if target_df is None:
                            st.error(f"  ⚠️ 타겟 변수 추출 실패 - 건너뜀")
                            continue
                    
                    elif file_type == "TYPE2":
                        st.warning("  ⚠️ TYPE2 파일 형식 - 건너뜀")
                        continue
                    
                    # Case 1 기본값 설정
                    if 'case 1' in sheet_name.lower():
                        st.session_state.default_values = input_vars.copy()
                        st.success(f"  ✅✅ Case 1 기본값으로 설정됨!")
                        for k, v in input_vars.items():
                            st.info(f"    {k} = {v:.2e}")
                    
                    # X 데이터 생성: [약물 변수 5개 + Time]
                    base_X = [input_vars.get(feat, 0.0) for feat in st.session_state.feature_names]
                    
                    samples_added = 0
                    for idx, row in target_df.iterrows():
                        # Time을 입력 변수에 추가
                        X_sample = base_X + [row['Time(h)']]
                        
                        # 🔥 올바른 순서: Total mass, Lymph, Blood, Decay, ECM
                        y_sample = [row['Total mass'], row['Lymph'], row['Blood'], 
                                   row['Decay'], row['ECM']]
                        
                        all_X.append(X_sample)
                        all_y.append(y_sample)
                        samples_added += 1
                    
                    st.success(f"  ✅ {samples_added}개 샘플 추가됨")
                    total_sheets_processed += 1
            
            st.success(f"✅ {uploaded_file.name} 완료")
            
        except Exception as e:
            st.error(f"❌ {uploaded_file.name} 처리 중 오류: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            continue
    
    if len(all_X) == 0:
        st.error("❌ 처리된 데이터가 없습니다.")
        return None, None
    
    X = np.array(all_X)
    y = np.array(all_y)
    
    st.success(f"🎉 총 {total_sheets_processed}개 시트에서 {len(X)}개 샘플 생성 완료!")
    st.info(f"📊 데이터셋 형태: X={X.shape}, y={y.shape}")
    st.info(f"📥 입력 변수: 약물 5개 + Time = 6개")
    st.info(f"📤 타겟 변수: {', '.join(st.session_state.target_names)}")
    st.info(f"⏰ Time 범위: {X[:, -1].min():.2f}h ~ {X[:, -1].max():.2f}h")
    
    return X, y

def train_xgboost_models(X, y, hyperparams):
    """XGBoost 모델 학습 - 각 타겟 변수별 독립 모델"""
    st.session_state.scaler_X = StandardScaler()
    X_scaled = st.session_state.scaler_X.fit_transform(X)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, target_name in enumerate(st.session_state.target_names):
        status_text.text(f"🎯 XGBoost 모델 학습 중: {target_name} ({idx+1}/{len(st.session_state.target_names)})")
        
        y_target = y[:, idx]
        
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1)).ravel()
        st.session_state.scaler_y[target_name] = scaler_y
        
        # XGBoost Regressor 생성
        xgb_model = xgb.XGBRegressor(
            n_estimators=hyperparams['n_estimators'],
            max_depth=hyperparams['max_depth'],
            learning_rate=hyperparams['learning_rate'],
            subsample=hyperparams['subsample'],
            colsample_bytree=hyperparams['colsample_bytree'],
            min_child_weight=hyperparams['min_child_weight'],
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(X_scaled, y_scaled)
        st.session_state.xgb_models[target_name] = xgb_model
        
        progress_bar.progress((idx + 1) / len(st.session_state.target_names))
    
    progress_bar.empty()
    status_text.empty()
    st.session_state.model_trained = True

def predict_time_series(input_values, time_points):
    """시계열 예측 - XGBoost 사용"""
    base_X = [input_values.get(feat, 0) for feat in st.session_state.feature_names]
    
    predictions_over_time = {target: [] for target in st.session_state.target_names}
    # XGBoost는 불확실성을 직접 제공하지 않으므로, 예측값만 반환
    uncertainties_over_time = {target: [] for target in st.session_state.target_names}
    
    for time_point in time_points:
        X_input = np.array([base_X + [time_point]])
        X_scaled = st.session_state.scaler_X.transform(X_input)
        
        for target_name in st.session_state.target_names:
            xgb_model = st.session_state.xgb_models[target_name]
            scaler_y = st.session_state.scaler_y[target_name]
            
            y_pred_scaled = xgb_model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
            
            # XGBoost는 불확실성을 직접 제공하지 않으므로, 간단한 추정값 사용
            # (실제로는 SHAP 값이나 다른 방법으로 불확실성을 추정할 수 있음)
            y_std = abs(y_pred) * 0.05  # 예측값의 5%를 불확실성으로 추정
            
            predictions_over_time[target_name].append(y_pred)
            uncertainties_over_time[target_name].append(y_std)
    
    return predictions_over_time, uncertainties_over_time

def smooth_curve(values, method='savgol', window_length=11, poly_order=3, spline_kind='cubic'):
    """
    곡선 스무딩 함수 (시각화용)
    
    Args:
        values: 스무딩할 값들의 배열
        method: 스무딩 방법 ('savgol', 'spline', 'moving_avg')
        window_length: Savitzky-Golay 필터의 윈도우 길이 (홀수여야 함)
        poly_order: 다항식 차수
        spline_kind: 스플라인 종류 ('linear', 'cubic', 'quadratic')
    
    Returns:
        스무딩된 값들의 배열
    """
    values = np.array(values)
    
    if len(values) < 3:
        return values
    
    if method == 'savgol':
        # Savitzky-Golay 필터
        # window_length는 홀수여야 하고, 데이터 길이보다 작아야 함
        wl = min(window_length, len(values))
        if wl % 2 == 0:
            wl -= 1
        if wl < 3:
            wl = 3
        
        po = min(poly_order, wl - 1)
        if po < 1:
            po = 1
        
        try:
            smoothed = savgol_filter(values, wl, po)
            # 음수 값 방지 (물리적 제약)
            smoothed = np.maximum(smoothed, 0)
            return smoothed
        except:
            return values
    
    elif method == 'spline':
        # 스플라인 보간
        x_original = np.arange(len(values))
        x_smooth = np.linspace(0, len(values) - 1, len(values) * 2)
        
        try:
            f = interp1d(x_original, values, kind=spline_kind, bounds_error=False, fill_value='extrapolate')
            smoothed = f(x_smooth)
            # 원래 길이로 다운샘플링
            indices = np.linspace(0, len(smoothed) - 1, len(values), dtype=int)
            smoothed = smoothed[indices]
            smoothed = np.maximum(smoothed, 0)
            return smoothed
        except:
            return values
    
    elif method == 'moving_avg':
        # 이동 평균
        window = min(window_length, len(values))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            window = 3
        
        # 패딩 추가 (경계 처리)
        padded = np.pad(values, (window // 2, window // 2), mode='edge')
        smoothed = np.convolve(padded, np.ones(window) / window, mode='valid')
        smoothed = np.maximum(smoothed, 0)
        return smoothed
    
    else:
        return values

def check_physical_validity(predictions):
    """물리적 유효성 검사"""
    warnings = []
    
    for key, values in predictions.items():
        if isinstance(values, list):
            if any(v < 0 for v in values):
                min_val = min(values)
                warnings.append(f"⚠️ {key}에 음수 값 존재 (최소: {min_val:.4e})")
        elif values < 0:
            warnings.append(f"⚠️ {key} 값이 음수입니다 ({values:.4e})")
    
    return warnings

# ==================== UI ====================

st.title("🧬 미세생리유체칩 약물 동태 분석 시스템")
st.markdown("### 디지털 트윈 시뮬레이션 및 XGBoost 회귀 모델 기반 예측")

# 사이드바
with st.sidebar:
    st.header("📁 데이터 업로드")
    
    uploaded_files = st.file_uploader(
        "엑셀 또는 CSV 파일",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="sol765.xlsx와 Injection site results.xlsx 모두 업로드하세요"
    )
    
    st.markdown("---")
    st.info(f"⏭️ 제외 시트: {', '.join(EXCLUDE_SHEETS)}")
    
    st.markdown("---")
    
    # 하이퍼파라미터 튜닝 UI
    st.header("⚙️ XGBoost 하이퍼파라미터")
    
    with st.expander("🔧 하이퍼파라미터 설정", expanded=False):
        st.session_state.hyperparams['n_estimators'] = st.slider(
            "n_estimators (트리 개수)",
            min_value=50,
            max_value=500,
            value=st.session_state.hyperparams['n_estimators'],
            step=50,
            help="더 많은 트리는 더 정확하지만 학습 시간이 길어집니다"
        )
        
        st.session_state.hyperparams['max_depth'] = st.slider(
            "max_depth (트리 깊이)",
            min_value=3,
            max_value=10,
            value=st.session_state.hyperparams['max_depth'],
            step=1,
            help="깊은 트리는 복잡한 패턴을 학습하지만 과적합 위험이 있습니다"
        )
        
        st.session_state.hyperparams['learning_rate'] = st.slider(
            "learning_rate (학습률)",
            min_value=0.01,
            max_value=0.3,
            value=st.session_state.hyperparams['learning_rate'],
            step=0.01,
            help="낮은 학습률은 더 안정적이지만 더 많은 트리가 필요합니다"
        )
        
        st.session_state.hyperparams['subsample'] = st.slider(
            "subsample (샘플 비율)",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.hyperparams['subsample'],
            step=0.1,
            help="각 트리에 사용할 샘플 비율 (과적합 방지)"
        )
        
        st.session_state.hyperparams['colsample_bytree'] = st.slider(
            "colsample_bytree (특성 샘플 비율)",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.hyperparams['colsample_bytree'],
            step=0.1,
            help="각 트리에 사용할 특성 비율"
        )
        
        st.session_state.hyperparams['min_child_weight'] = st.slider(
            "min_child_weight (최소 자식 가중치)",
            min_value=1,
            max_value=10,
            value=st.session_state.hyperparams['min_child_weight'],
            step=1,
            help="리프 노드의 최소 샘플 수 (과적합 방지)"
        )
    
    st.markdown("---")
    
    if uploaded_files:
        if st.button("🚀 데이터 학습 시작", type="primary", use_container_width=True):
            with st.spinner("처리 중..."):
                X, y = load_and_process_files(uploaded_files)
                
                if X is not None and y is not None:
                    st.markdown("---")
                    st.info(f"📊 총 샘플: {X.shape[0]}개")
                    st.info(f"📥 입력: 약물 5개 + Time")
                    st.info(f"📤 타겟: {y.shape[1]}개")
                    
                    train_xgboost_models(X, y, st.session_state.hyperparams)
                    st.success("✅✅ XGBoost 모델 학습 완료!")
                    
                    if st.session_state.default_values:
                        st.markdown("---")
                        st.success("🎯 Case 1 기본값:")
                        for feat in st.session_state.feature_names:
                            val = st.session_state.default_values.get(feat, 0)
                            st.text(f"{feat}: {val:.2e}")
                else:
                    st.error("❌ 데이터 추출 실패")
    
    st.markdown("---")
    
    # 그래프 스무딩 옵션 UI
    st.header("📈 그래프 스무딩")
    
    st.session_state.smoothing_params['enabled'] = st.checkbox(
        "스무딩 활성화",
        value=st.session_state.smoothing_params['enabled'],
        help="그래프를 매끄럽게 표시합니다 (원본 예측값은 변경되지 않습니다)"
    )
    
    if st.session_state.smoothing_params['enabled']:
        st.session_state.smoothing_params['method'] = st.selectbox(
            "스무딩 방법",
            ['savgol', 'spline', 'moving_avg'],
            index=['savgol', 'spline', 'moving_avg'].index(st.session_state.smoothing_params['method']),
            help="Savitzky-Golay: 노이즈 제거에 효과적 | Spline: 부드러운 곡선 | Moving Avg: 간단한 평활화"
        )
        
        if st.session_state.smoothing_params['method'] == 'savgol':
            st.session_state.smoothing_params['window_length'] = st.slider(
                "윈도우 길이 (홀수)",
                min_value=5,
                max_value=51,
                value=st.session_state.smoothing_params['window_length'],
                step=2,
                help="값이 클수록 더 부드럽지만 세부 특징이 사라질 수 있습니다"
            )
            st.session_state.smoothing_params['poly_order'] = st.slider(
                "다항식 차수",
                min_value=1,
                max_value=5,
                value=st.session_state.smoothing_params['poly_order'],
                help="윈도우 길이보다 작아야 합니다"
            )
        
        elif st.session_state.smoothing_params['method'] == 'spline':
            st.session_state.smoothing_params['spline_kind'] = st.selectbox(
                "스플라인 종류",
                ['linear', 'quadratic', 'cubic'],
                index=['linear', 'quadratic', 'cubic'].index(st.session_state.smoothing_params['spline_kind']),
                help="cubic이 가장 부드럽습니다"
            )
        
        elif st.session_state.smoothing_params['method'] == 'moving_avg':
            st.session_state.smoothing_params['window_length'] = st.slider(
                "윈도우 길이 (홀수)",
                min_value=3,
                max_value=21,
                value=st.session_state.smoothing_params['window_length'],
                step=2,
                help="평균을 낼 데이터 포인트 수"
            )
    
    st.markdown("---")
    st.header("ℹ️ 데이터 순서")
    st.markdown("""
    **올바른 타겟 순서:**
    1. Total mass
    2. Lymph
    3. Blood
    4. **Decay** ← 4번째!
    5. **ECM** ← 5번째!
    
    (이전 버전은 순서가 바뀌어 있었음)
    """)

# 메인 영역
if st.session_state.model_trained:
    st.success("✅ 모델 학습 완료. 시계열 예측을 진행할 수 있습니다.")
    
    st.markdown("---")
    st.header("🔮 약물 동태 예측")
    
    input_method = st.radio(
        "입력 방식",
        ["기본값 사용 (Case 1)", "직접 입력"],
        horizontal=True
    )
    
    input_values = {}
    
    if input_method == "기본값 사용 (Case 1)":
        if st.session_state.default_values:
            input_values = st.session_state.default_values.copy()
            st.info("📋 Case 1 기본값 사용")
            
            cols = st.columns(5)
            for idx, feat in enumerate(st.session_state.feature_names):
                with cols[idx]:
                    value = input_values.get(feat, 0.0)
                    st.metric(feat, f"{value:.2e}")
        else:
            st.warning("⚠️ Case 1 파일 없음. 직접 입력하세요.")
            input_method = "직접 입력"
    
    if input_method == "직접 입력":
        st.markdown("**입력 변수를 직접 입력해주세요:**")
        
        cols = st.columns(5)
        for idx, feat in enumerate(st.session_state.feature_names):
            with cols[idx]:
                default_val = 0.0
                if st.session_state.default_values:
                    default_val = st.session_state.default_values.get(feat, 0.0)
                
                input_values[feat] = st.number_input(
                    feat,
                    value=float(default_val),
                    format="%.2e",
                    key=f"input_{feat}"
                )
    
    num_points = st.slider("시계열 예측 포인트 수", min_value=50, max_value=200, value=100, step=10)
    
    if st.button("🎯 예측 실행 (0-72시간)", type="primary", use_container_width=True):
        with st.spinner("0~72시간 시계열 예측 수행 중..."):
            time_points = np.linspace(0, 72, num_points)
            predictions, uncertainties = predict_time_series(input_values, time_points)
            
            validity_warnings = check_physical_validity(predictions)
            for warning in validity_warnings:
                st.warning(warning)
            
            st.markdown("---")
            st.header("📊 예측 결과")
            
            st.markdown("### 📈 시간에 따른 약물 동태 변화 (0-72시간)")
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # 🔥 올바른 색상 매핑 (Decay가 파랑, ECM이 주황)
            colors = {
                'Total mass': '#000000',
                'Lymph': '#00FF00',
                'Blood': '#FF0000',
                'Decay': '#0000FF',  # Decay = 파랑
                'ECM': '#FFA500'     # ECM = 주황
            }
            
            for target_name in st.session_state.target_names:
                color = colors.get(target_name, '#888888')
                pred_values = predictions[target_name]
                uncert_values = uncertainties[target_name]
                
                # 스무딩 적용 (시각화용만, 원본 값은 유지)
                if st.session_state.smoothing_params['enabled']:
                    smoothed_values = smooth_curve(
                        pred_values,
                        method=st.session_state.smoothing_params['method'],
                        window_length=st.session_state.smoothing_params['window_length'],
                        poly_order=st.session_state.smoothing_params.get('poly_order', 3),
                        spline_kind=st.session_state.smoothing_params.get('spline_kind', 'cubic')
                    )
                    # 그래프에는 스무딩된 값 사용
                    plot_values = smoothed_values
                else:
                    # 스무딩 비활성화 시 원본 값 사용
                    plot_values = pred_values
                
                ax.plot(time_points, plot_values, color=color, 
                       linewidth=2.5, label=target_name)
                
                # 불확실성 영역은 원본 예측값 기준으로 표시
                pred_array = np.array(pred_values)
                uncert_array = np.array(uncert_values)
                ax.fill_between(time_points,
                               pred_array - 1.96 * uncert_array,
                               pred_array + 1.96 * uncert_array,
                               color=color, alpha=0.1)
            
            ax.set_xlabel('Time (h)', fontsize=14, fontweight='bold')
            ax.set_ylabel('%Mass (m/m₀)', fontsize=14, fontweight='bold')
            ax.set_title('Representative', fontsize=16, fontweight='bold', style='italic')
            ax.legend(loc='center right', fontsize=12, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(0, 72)
            ax.set_ylim(0, 105)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 주요 시간대
            st.markdown("---")
            st.markdown("### 📋 주요 시간대 예측값")
            
            key_times = [0, 6, 12, 24, 48, 72]
            key_indices = [np.argmin(np.abs(time_points - t)) for t in key_times]
            
            table_data = {'Time (h)': [time_points[i] for i in key_indices]}
            for target_name in st.session_state.target_names:
                table_data[target_name] = [f"{predictions[target_name][i]:.2f}" 
                                          for i in key_indices]
            
            df_display = pd.DataFrame(table_data)
            st.dataframe(df_display, use_container_width=True)
            
            # CSV 다운로드
            st.markdown("---")
            full_data = {'Time (h)': time_points}
            for target_name in st.session_state.target_names:
                full_data[target_name] = predictions[target_name]
                full_data[f'{target_name}_uncertainty'] = uncertainties[target_name]
            
            df_full = pd.DataFrame(full_data)
            csv = df_full.to_csv(index=False)
            
            st.download_button(
                label="📥 전체 예측 데이터 CSV 다운로드",
                data=csv,
                file_name="prediction_timeseries.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # 입력값
            st.markdown("---")
            st.markdown("### 🔧 사용된 입력 변수")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**약물 파라미터**")
                input_df = pd.DataFrame({
                    '변수명': list(input_values.keys()),
                    '입력값': [f"{v:.6e}" for v in input_values.values()]
                })
                st.dataframe(input_df, use_container_width=True)
            
            with col2:
                st.markdown("**예측 설정**")
                st.text(f"시간 범위: 0-72 시간")
                st.text(f"예측 포인트: {num_points}개")
                st.text(f"간격: {72/num_points:.3f} 시간")

else:
    st.info("👈 사이드바에서 데이터를 업로드하고 학습하세요")
    
    st.markdown("---")
    st.header("📖 사용 방법")
    
    with st.expander("🔥 XGBoost 모델 정보", expanded=True):
        st.markdown("""
        ### 모델 특징
        
        **1. XGBoost 회귀 모델**
        - Gaussian Process 대신 XGBoost 사용
        - 각 타겟 변수별 독립적인 모델 학습
        - 더 빠른 학습 및 예측 속도
        
        **2. 하이퍼파라미터 튜닝**
        - n_estimators: 트리 개수 조정
        - max_depth: 트리 깊이 조정
        - learning_rate: 학습률 조정
        - subsample, colsample_bytree: 과적합 방지
        
        **3. 타겟 변수 순서**
        - Total mass → Lymph → Blood → **Decay** → **ECM**
        
        **4. 색상 매핑**
        - Decay = 파랑
        - ECM = 주황
        """)
    
    with st.expander("💡 사용 팁"):
        st.markdown("""
        1. 두 파일 모두 업로드
        2. "학습 시작" 클릭
        3. 입력 변수 설정
        4. "예측 실행"
        5. 그래프가 부드러운 곡선으로 나와야 정상
        """)

st.markdown("---")
st.caption("🧬 미세생리유체칩 디지털 트윈 시뮬레이션 | XGBoost 회귀 모델")
