# ============================================================
# 1. 필요한 라이브러리 임포트
# ============================================================
# 기본 라이브러리
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# LangChain 라이브러리 (선택적 임포트)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain 라이브러리 로드 성공")
except ImportError:
    print("⚠️ LangChain 라이브러리가 설치되지 않았습니다. 기본 분석 기능만 사용됩니다.")
    LANGCHAIN_AVAILABLE = False

load_dotenv(override=True)
print("✅ 환경 변수 로드 완료")

# ============================================================
# 2. Pydantic 모델 정의 (구조화된 출력)
# ============================================================
class BloodPressureInsight(BaseModel):
    """개별 환자 혈압 분석 결과 구조"""
    overall_assessment: str = Field(description="전반적인 혈압 상태 평가")
    risk_level: str = Field(description="위험도 수준")
    key_risk_factors: List[str] = Field(description="주요 위험 요인들")
    recommendations: List[str] = Field(description="개선 권장사항")
    lifestyle_advice: str = Field(description="생활습관 조언")
    follow_up_needed: bool = Field(description="추가 검진 필요 여부")


class DatasetInsight(BaseModel):
    """데이터셋 전체 분석 결과 구조"""
    summary: str = Field(description="데이터셋 전반적 요약")
    key_patterns: List[str] = Field(description="주요 발견된 패턴들")
    statistical_highlights: List[str] = Field(description="통계적 주요점")
    clinical_implications: List[str] = Field(description="임상적 의미")

print("✅ Pydantic 모델 정의 완료")

# ============================================================
# 3. LangChainBPProcessor 클래스 정의
# ============================================================
class LangChainBPProcessor:
    """
    LangChain을 활용한 혈압 데이터 AI 처리기
    
    【핵심 기능】
    1. 개별 환자 혈압 분석 (AI 기반)
    2. 데이터셋 전체 인사이트 도출
    3. 개인 맞춤 건강 조언 생성
    4. AI 사용 불가 시 폴백(fallback) 기능
    
    【디자인 패턴】
    - Strategy Pattern: AI/전통적 알고리즘 선택적 사용
    - Graceful Degradation: AI 실패 시 기본 기능 제공
    """
    # ============================================================
    # 3-1. 생성자
    # ============================================================    
    def __init__(self, api_key: Optional[str] = None):
        """
        LangChain 처리기 초기화
        
        【매개변수】
        api_key (Optional[str]): OpenAI API 키
            - 직접 전달하거나
            - .env 파일의 OPENAI_API_KEY 사용
        
        【초기화 과정】
        1. API 키 확인
        2. ChatOpenAI 객체 생성
        3. 연결 테스트
        """
        # API 키 우선순위: 인자 > 환경변수
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.llm = None  # LLM 객체 초기값
        
        # LangChain이 사용 가능하고 API 키가 있는 경우
        if LANGCHAIN_AVAILABLE and self.api_key:
            try:
                # ChatOpenAI 초기화
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",  # 사용 모델
                    temperature=0.1,  # 낮은 값 = 일관성 높음 (0~1)
                    api_key=self.api_key
                )
                print("✅ LangChain AI 시스템 초기화 완료")
            except Exception as e:
                print(f"⚠️ OpenAI API 연결 실패: {e}")
                self.llm = None
        else:
            if not self.api_key:
                print("⚠️ OpenAI API 키가 없습니다. .env 파일에 OPENAI_API_KEY를 설정하세요.")
        # ============================================================
    # 3-2. 개별 환자의 ai 종합 분석
    # ============================================================
    def analyze_individual_bp(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """개별 환자의 혈압 및 건강 데이터를 AI로 종합 분석"""
        
        if not self.llm:
            return self._fallback_individual_analysis(patient_data)
        
        try:
            # Structured Output을 지원하는 모델로 변경
            
            
            structured_llm = ChatOpenAI(
                model="gpt-4o-mini",  # 또는 "gpt-4o-2024-08-06"
                temperature=0.1,
                api_key=self.api_key
            ).with_structured_output(BloodPressureInsight)
            
            # 프롬프트 생성
            patient_info = self._format_patient_info(patient_data)
            
            messages = [
                SystemMessage(content="""당신은 심혈관 질환 전문의입니다.
    환자의 혈압 및 관련 데이터를 종합적으로 분석하여 전문적인 의학적 조언을 제공합니다.

    분석 시 고려사항:
    1. 대한고혈압학회 및 미국심장학회(AHA) 가이드라인 준수
    2. 환자의 개별적 위험 요인 종합 평가
    3. 실용적이고 구체적인 생활습관 개선 방안 제시
    4. 의료진 상담이 필요한 경우 명확히 권고

    ⚠️ 주의: 이는 교육 및 참고 목적이며, 실제 의료 진단이나 치료를 대체할 수 없습니다."""),
                HumanMessage(content=f"환자 정보:\n{patient_info}\n\n위 환자 정보를 분석하여 결과를 제공해주세요.")
            ]
            
            # AI 호출 - 자동으로 BloodPressureInsight 형식으로 반환
            result = structured_llm.invoke(messages)
            
            return {
                'analysis_type': 'AI_분석',
                'timestamp': datetime.now().isoformat(),
                'overall_assessment': result.overall_assessment,
                'risk_level': result.risk_level,
                'key_risk_factors': result.key_risk_factors,
                'recommendations': result.recommendations,
                'lifestyle_advice': result.lifestyle_advice,
                'follow_up_needed': result.follow_up_needed,
                'source': 'GPT-4o-mini'
            }
            
        except Exception as e:
            print(f"⚠️ AI 분석 중 오류 발생: {e}")
            return self._fallback_individual_analysis(patient_data)

		# ============================================================
    # 3-3. 데이터셋 전체 인사이트 분석
    # ============================================================
    def analyze_dataset_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        전체 데이터셋에 대한 AI 인사이트 도출
        
        【매개변수】
        df (DataFrame): 환자 데이터프레임
        
        【반환값】
        Dict: 데이터셋 분석 결과
        
        【분석 목적】
        개별 환자가 아닌 전체 집단의 패턴을 발견하여
        공중보건학적 인사이트 도출
        
        【AI의 강점 활용】
        - 복잡한 패턴 인식
        - 다차원 상관관계 해석
        - 임상적 의미 도출
        """
        
        if not self.llm:
            return self._fallback_dataset_analysis(df)
        
        try:
            # --------------------------------------------------
            # ✅ Structured Output을 지원하는 모델로 변경
            # --------------------------------------------------
            from langchain_openai import ChatOpenAI
            
            structured_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=self.api_key
            ).with_structured_output(DatasetInsight)
            
            # --------------------------------------------------
            # 데이터셋 기본 통계 생성
            # --------------------------------------------------
            dataset_summary = self._generate_dataset_summary(df)
            
            # --------------------------------------------------
            # ✅ 프롬프트 메시지 (format_instructions 제거)
            # --------------------------------------------------
            messages = [
                SystemMessage(content="""당신은 의료 데이터 분석 전문가입니다.
    PhysioNet 혈압 데이터셋을 분석하여 의미있는 인사이트와 패턴을 발견합니다.

    분석 관점:
    1. 연령, 성별, BMI별 혈압 패턴
    2. 생활습관 요인들의 영향
    3. 가족력과 혈압의 관계
    4. 임상적으로 중요한 발견사항
    5. 공중보건학적 시사점

    결과를 구조화된 형식으로 제공해주세요."""),
                HumanMessage(content=f"데이터셋 요약:\n{dataset_summary}\n\n위 데이터를 분석하여 인사이트를 제공해주세요.")
            ]
            
            # --------------------------------------------------
            # ✅ AI 호출 - 자동으로 DatasetInsight 형식으로 반환
            # --------------------------------------------------
            result = structured_llm.invoke(messages)
            
            return {
                'analysis_type': 'Dataset_AI_분석',
                'timestamp': datetime.now().isoformat(),
                'total_patients': len(df),
                'summary': result.summary,
                'key_patterns': result.key_patterns,
                'statistical_highlights': result.statistical_highlights,
                'clinical_implications': result.clinical_implications,
                'source': 'GPT-4o-mini'
            }
            
        except Exception as e:
            print(f"⚠️ 데이터셋 AI 분석 중 오류 발생: {e}")
            return self._fallback_dataset_analysis(df)            

    # ============================================================
    # 3-4. 개인 맞춤 건강 조언 생성
    # ============================================================
    def generate_health_advice(self, 
                             patient_data: Dict[str, Any],
                             prediction_result: Dict[str, Any]) -> str:
        """
        예측 결과를 바탕으로 개인 맞춤 건강 조언 생성
        
        【매개변수】
        patient_data (Dict): 환자 정보
        prediction_result (Dict): ML 모델의 혈압 예측 결과
        
        【반환값】
        str: 자연어로 작성된 건강 조언
        
        【AI 활용 이유】
        - 전문적이면서도 이해하기 쉬운 표현
        - 환자 상황에 맞는 맞춤형 메시지
        - 긍정적이고 격려하는 톤
        
        【프롬프트 디자인】
        - Tone Setting: "친근한 건강 상담사"
        - Output Format: 3-5개 핵심 포인트
        - Safety: 의료진 상담 권고 포함
        """
        
        if not self.llm:
            return self._fallback_health_advice(patient_data, prediction_result)
        
        try:
            # --------------------------------------------------
            # 3-4-1 프롬프트 작성 (구조화된 출력 없이 자유 형식)
            # --------------------------------------------------
            messages = [
                SystemMessage(content="""당신은 친근한 건강 상담사입니다.
                    환자의 정보와 혈압 예측 결과를 바탕으로 따뜻하고 실용적인 건강 조언을 제공합니다.

                    조언 형식:
                    - 이해하기 쉬운 한국어 사용
                    - 구체적이고 실행 가능한 제안
                    - 긍정적이고 격려하는 톤
                    - 3-5개의 핵심 포인트로 정리

                    ⚠️ 의료진 상담이 필요한 경우 반드시 권고"""),
                                    
                                    HumanMessage(content=f"""환자 정보:
                    - 나이: {patient_data.get('age', 'N/A')}세
                    - 성별: {patient_data.get('gender', 'N/A')}
                    - BMI: {patient_data.get('bmi', 'N/A')}
                    - 흡연: {'예' if patient_data.get('smoking', 0) else '아니오'}
                    - 운동 빈도: 주 {patient_data.get('exercise_frequency', 0)}회
                    - 스트레스 수준: {patient_data.get('stress_level', 'N/A')}/10

                    예측된 혈압:
                    - 수축기: {prediction_result.get('systolic_bp', 'N/A'):.1f} mmHg
                    - 이완기: {prediction_result.get('diastolic_bp', 'N/A'):.1f} mmHg

                    개인 맞춤 건강 조언을 작성해 주세요.""")
            ]
            
            # --------------------------------------------------
            # 3-4-2 AI 호출 및 결과 반환
            # --------------------------------------------------
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"⚠️ 건강 조언 생성 중 오류: {e}")
            return self._fallback_health_advice(patient_data, prediction_result)

    # ============================================================
    # 3-5. 헬퍼 메서드들
    # ============================================================
    
    def _format_patient_info(self, patient_data: Dict[str, Any]) -> str:
        """
        환자 정보를 AI가 이해하기 쉬운 자연어 텍스트로 변환
        
        【목적】
        딕셔너리 → 구조화된 텍스트 변환
        
        【변환 예시】
        {'age': 45, 'gender': '남성', 'bmi': 26.1}
        →
        "나이: 45세
         성별: 남성
         BMI: 26.1"
        """
        info_lines = []
        
        # 기본 정보
        if 'age' in patient_data:
            info_lines.append(f"나이: {patient_data['age']}세")
        if 'gender' in patient_data:
            info_lines.append(f"성별: {patient_data['gender']}")
        if 'bmi' in patient_data:
            info_lines.append(f"BMI: {patient_data['bmi']}")
        
        # 생활습관
        if 'smoking' in patient_data:
            smoking_status = "흡연자" if patient_data['smoking'] else "비흡연자"
            info_lines.append(f"흡연 상태: {smoking_status}")
        if 'exercise_frequency' in patient_data:
            info_lines.append(f"운동 빈도: 주 {patient_data['exercise_frequency']}회")
        if 'stress_level' in patient_data:
            info_lines.append(f"스트레스 수준: {patient_data['stress_level']}/10")
        
        # 건강 지표
        if 'heart_rate_bpm' in patient_data:
            info_lines.append(f"심박수: {patient_data['heart_rate_bpm']} bpm")
        if 'systolic_bp' in patient_data and 'diastolic_bp' in patient_data:
            info_lines.append(f"현재 혈압: {patient_data['systolic_bp']}/{patient_data['diastolic_bp']} mmHg")
        
        # 가족력
        family_history = []
        if patient_data.get('family_history_hypertension'):
            family_history.append("고혈압")
        if patient_data.get('family_history_diabetes'):
            family_history.append("당뇨병")
        if family_history:
            info_lines.append(f"가족력: {', '.join(family_history)}")
        
        return "\n".join(info_lines)
    
    
    def _generate_dataset_summary(self, df: pd.DataFrame) -> str:
        """
        데이터프레임의 주요 통계를 텍스트로 요약
        
        【목적】
        AI에게 데이터셋의 전반적인 특성을 전달
        
        【포함 정보】
        - 기본 통계 (평균, 범위)
        - 분포 정보 (성별, 혈압 분류)
        - 주요 비율 (흡연율, 고혈압 유병률)
        """
        summary_lines = []
        
        # 기본 정보
        summary_lines.append(f"총 환자 수: {len(df)}")
        
        # 연령 분포
        if 'age' in df.columns:
            summary_lines.append(
                f"연령 범위: {df['age'].min()}-{df['age'].max()}세 "
                f"(평균 {df['age'].mean():.1f}세)"
            )
        
        # 성별 분포
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            summary_lines.append(f"성별 분포: {gender_counts.to_dict()}")
        
        # 혈압 통계
        if 'systolic_bp' in df.columns:
            summary_lines.append(
                f"수축기 혈압 범위: {df['systolic_bp'].min()}-{df['systolic_bp'].max()} mmHg "
                f"(평균 {df['systolic_bp'].mean():.1f})"
            )
        
        # BMI 통계
        if 'bmi' in df.columns:
            summary_lines.append(
                f"BMI 범위: {df['bmi'].min():.1f}-{df['bmi'].max():.1f} "
                f"(평균 {df['bmi'].mean():.1f})"
            )
        
        # 흡연율
        if 'smoking' in df.columns:
            smoking_rate = (df['smoking'].sum() / len(df)) * 100
            summary_lines.append(f"흡연율: {smoking_rate:.1f}%")
        
        # 혈압 분류
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            hypertension = ((df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)).sum()
            prehypertension = (
                ((df['systolic_bp'] >= 120) & (df['systolic_bp'] < 140)) |
                ((df['diastolic_bp'] >= 80) & (df['diastolic_bp'] < 90))
            ).sum()
            normal = len(df) - hypertension - prehypertension
            summary_lines.append(
                f"혈압 분류 - 정상: {normal}명, "
                f"고혈압전단계: {prehypertension}명, "
                f"고혈압: {hypertension}명"
            )
        
        return "\n".join(summary_lines)            

    # ============================================================
    # 3-5. 헬퍼 메서드들
    # ============================================================
    
    def _format_patient_info(self, patient_data: Dict[str, Any]) -> str:
        """
        환자 정보를 AI가 이해하기 쉬운 자연어 텍스트로 변환
        
        【목적】
        딕셔너리 → 구조화된 텍스트 변환
        
        【변환 예시】
        {'age': 45, 'gender': '남성', 'bmi': 26.1}
        →
        "나이: 45세
         성별: 남성
         BMI: 26.1"
        """
        info_lines = []
        
        # 기본 정보
        if 'age' in patient_data:
            info_lines.append(f"나이: {patient_data['age']}세")
        if 'gender' in patient_data:
            info_lines.append(f"성별: {patient_data['gender']}")
        if 'bmi' in patient_data:
            info_lines.append(f"BMI: {patient_data['bmi']}")
        
        # 생활습관
        if 'smoking' in patient_data:
            smoking_status = "흡연자" if patient_data['smoking'] else "비흡연자"
            info_lines.append(f"흡연 상태: {smoking_status}")
        if 'exercise_frequency' in patient_data:
            info_lines.append(f"운동 빈도: 주 {patient_data['exercise_frequency']}회")
        if 'stress_level' in patient_data:
            info_lines.append(f"스트레스 수준: {patient_data['stress_level']}/10")
        
        # 건강 지표
        if 'heart_rate_bpm' in patient_data:
            info_lines.append(f"심박수: {patient_data['heart_rate_bpm']} bpm")
        if 'systolic_bp' in patient_data and 'diastolic_bp' in patient_data:
            info_lines.append(f"현재 혈압: {patient_data['systolic_bp']}/{patient_data['diastolic_bp']} mmHg")
        
        # 가족력
        family_history = []
        if patient_data.get('family_history_hypertension'):
            family_history.append("고혈압")
        if patient_data.get('family_history_diabetes'):
            family_history.append("당뇨병")
        if family_history:
            info_lines.append(f"가족력: {', '.join(family_history)}")
        
        return "\n".join(info_lines)
    
    
    def _generate_dataset_summary(self, df: pd.DataFrame) -> str:
        """
        데이터프레임의 주요 통계를 텍스트로 요약
        
        【목적】
        AI에게 데이터셋의 전반적인 특성을 전달
        
        【포함 정보】
        - 기본 통계 (평균, 범위)
        - 분포 정보 (성별, 혈압 분류)
        - 주요 비율 (흡연율, 고혈압 유병률)
        """
        summary_lines = []
        
        # 기본 정보
        summary_lines.append(f"총 환자 수: {len(df)}")
        
        # 연령 분포
        if 'age' in df.columns:
            summary_lines.append(
                f"연령 범위: {df['age'].min()}-{df['age'].max()}세 "
                f"(평균 {df['age'].mean():.1f}세)"
            )
        
        # 성별 분포
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            summary_lines.append(f"성별 분포: {gender_counts.to_dict()}")
        
        # 혈압 통계
        if 'systolic_bp' in df.columns:
            summary_lines.append(
                f"수축기 혈압 범위: {df['systolic_bp'].min()}-{df['systolic_bp'].max()} mmHg "
                f"(평균 {df['systolic_bp'].mean():.1f})"
            )
        
        # BMI 통계
        if 'bmi' in df.columns:
            summary_lines.append(
                f"BMI 범위: {df['bmi'].min():.1f}-{df['bmi'].max():.1f} "
                f"(평균 {df['bmi'].mean():.1f})"
            )
        
        # 흡연율
        if 'smoking' in df.columns:
            smoking_rate = (df['smoking'].sum() / len(df)) * 100
            summary_lines.append(f"흡연율: {smoking_rate:.1f}%")
        
        # 혈압 분류
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            hypertension = ((df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)).sum()
            prehypertension = (
                ((df['systolic_bp'] >= 120) & (df['systolic_bp'] < 140)) |
                ((df['diastolic_bp'] >= 80) & (df['diastolic_bp'] < 90))
            ).sum()
            normal = len(df) - hypertension - prehypertension
            summary_lines.append(
                f"혈압 분류 - 정상: {normal}명, "
                f"고혈압전단계: {prehypertension}명, "
                f"고혈압: {hypertension}명"
            )
        
        return "\n".join(summary_lines)            

    # ============================================================
    # 3-6. Fallback 메서드들 (AI 사용 불가 시)
    # ============================================================
    """
    Graceful Degradation 패턴:
    AI가 사용 불가능해도 기본적인 기능은 제공
    """
    
    def _fallback_individual_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI 사용 불가 시 전통적 알고리즘을 사용한 개별 분석
        
        【알고리즘】
        1. 위험 점수 계산 (0-10점)
        2. 각 위험 요인에 가중치 부여
        3. 점수에 따라 위험도 분류
        4. 규칙 기반 권장사항 생성
        """
        age = patient_data.get('age', 0)
        systolic = patient_data.get('systolic_bp', 120)
        diastolic = patient_data.get('diastolic_bp', 80)
        bmi = patient_data.get('bmi', 23)
        
        # 위험도 점수 계산
        risk_score = 0
        risk_factors = []
        
        # 혈압 수치 평가
        if systolic >= 140 or diastolic >= 90:
            risk_score += 3
            risk_factors.append("고혈압 수치")
        elif systolic >= 120 or diastolic >= 80:
            risk_score += 1
            risk_factors.append("혈압 경계수치")
        
        # 연령 평가
        if age >= 65:
            risk_score += 2
            risk_factors.append("고령")
        elif age >= 45:
            risk_score += 1
            risk_factors.append("중년")
        
        # BMI 평가
        if bmi >= 30:
            risk_score += 2
            risk_factors.append("비만")
        elif bmi >= 25:
            risk_score += 1
            risk_factors.append("과체중")
        
        # 흡연 평가
        if patient_data.get('smoking', 0):
            risk_score += 2
            risk_factors.append("흡연")
        
        # 가족력 평가
        if patient_data.get('family_history_hypertension', 0):
            risk_score += 1
            risk_factors.append("고혈압 가족력")
        
        # 위험도 분류
        if risk_score >= 6:
            risk_level = "매우높음"
        elif risk_score >= 4:
            risk_level = "높음"
        elif risk_score >= 2:
            risk_level = "보통"
        else:
            risk_level = "낮음"
        
        # 권장사항 생성
        recommendations = []
        if systolic >= 140:
            recommendations.append("의료진 상담을 통한 혈압 관리")
        if bmi >= 25:
            recommendations.append("체중 감량을 통한 BMI 정상화")
        if patient_data.get('exercise_frequency', 0) < 3:
            recommendations.append("주 3회 이상 규칙적인 유산소 운동")
        if patient_data.get('smoking', 0):
            recommendations.append("금연")
        recommendations.append("저나트륨 식단 실천")
        recommendations.append("충분한 수면과 스트레스 관리")
        
        return {
            'analysis_type': '기본_분석',
            'timestamp': datetime.now().isoformat(),
            'overall_assessment': f"현재 혈압 수치 {systolic}/{diastolic} mmHg로 {risk_level} 위험군에 속합니다.",
            'risk_level': risk_level,
            'key_risk_factors': risk_factors,
            'recommendations': recommendations,
            'lifestyle_advice': "규칙적인 운동, 건강한 식단, 스트레스 관리가 혈압 개선에 도움이 됩니다.",
            'follow_up_needed': risk_score >= 4,
            'source': '기본_알고리즘'
        }
    
    
    def _fallback_dataset_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        AI 사용 불가 시 전통적 통계 분석을 사용한 데이터셋 분석
        
        【알고리즘】
        1. 기본 상관관계 분석
        2. 그룹별 평균 비교
        3. 유병률 계산
        4. 규칙 기반 인사이트 도출
        """
        patterns = []
        highlights = []
        implications = []
        
        # 상관관계 분석
        if 'age' in df.columns and 'systolic_bp' in df.columns:
            age_bp_corr = df['age'].corr(df['systolic_bp'])
            patterns.append(f"연령과 수축기 혈압 상관관계: {age_bp_corr:.3f}")
        
        if 'bmi' in df.columns and 'systolic_bp' in df.columns:
            bmi_bp_corr = df['bmi'].corr(df['systolic_bp'])
            patterns.append(f"BMI와 수축기 혈압 상관관계: {bmi_bp_corr:.3f}")
        
        # 성별 분석
        if 'gender' in df.columns and 'systolic_bp' in df.columns:
            gender_bp = df.groupby('gender')['systolic_bp'].mean()
            patterns.append(f"성별 평균 수축기 혈압 차이 확인됨")
        
        # 고혈압 유병률
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            hypertension_rate = ((df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)).mean() * 100
            highlights.append(f"고혈압 유병률: {hypertension_rate:.1f}%")
        
        # 흡연율
        if 'smoking' in df.columns:
            smoking_rate = df['smoking'].mean() * 100
            highlights.append(f"흡연율: {smoking_rate:.1f}%")
        
        # 임상적 의미
        implications.append("연령 증가에 따른 혈압 상승 패턴 확인")
        implications.append("생활습관 개선을 통한 혈압 관리 가능성")
        implications.append("개별 맞춤형 위험도 평가 필요성")
        
        return {
            'analysis_type': 'Dataset_기본_분석',
            'timestamp': datetime.now().isoformat(),
            'total_patients': len(df),
            'summary': f"{len(df)}명의 환자 데이터를 분석하여 혈압과 관련 요인들 간의 관계를 파악했습니다.",
            'key_patterns': patterns,
            'statistical_highlights': highlights,
            'clinical_implications': implications,
            'source': '기본_통계분석'
        }
    
    
    def _fallback_health_advice(self, 
                               patient_data: Dict[str, Any],
                               prediction_result: Dict[str, Any]) -> str:
        """
        AI 사용 불가 시 규칙 기반 건강 조언 생성
        
        【알고리즘】
        1. 혈압 수치에 따른 기본 조언
        2. 개별 위험 요인별 맞춤 조언
        3. 일반적인 건강 수칙 추가
        """
        advice_lines = []
        systolic = prediction_result.get('systolic_bp', 120)
        age = patient_data.get('age', 0)
        bmi = patient_data.get('bmi', 23)
        
        advice_lines.append(f"예측된 혈압 {systolic:.1f} mmHg에 따른 건강 관리 방안:")
        advice_lines.append("")
        
        # 혈압별 조언
        if systolic >= 140:
            advice_lines.append("🚨 고혈압 수치입니다. 즉시 의료진 상담을 받으시기 바랍니다.")
        elif systolic >= 120:
            advice_lines.append("⚠️ 고혈압 전단계입니다. 생활습관 개선이 중요합니다.")
        else:
            advice_lines.append("✅ 정상 혈압 범위입니다. 현재 상태를 유지하세요.")
        
        advice_lines.append("")
        
        # 개별 맞춤 조언
        if bmi >= 25:
            advice_lines.append("• 체중 감량을 통해 혈압을 5-10 mmHg 낮출 수 있습니다.")
        if patient_data.get('exercise_frequency', 0) < 3:
            advice_lines.append("• 주 3-4회, 30분 이상 유산소 운동을 시작해보세요.")
        if patient_data.get('smoking', 0):
            advice_lines.append("• 금연은 즉시 심혈관 건강 개선에 도움이 됩니다.")
        
        advice_lines.append("• 하루 나트륨 섭취량을 2,300mg 미만으로 제한하세요.")
        advice_lines.append("• 충분한 수면(7-8시간)과 스트레스 관리가 중요합니다.")
        
        if systolic >= 140:
            advice_lines.append("")
            advice_lines.append("⚠️ 주의: 이 분석은 교육 목적이며, 실제 의료 진단을 대체할 수 없습니다.")
        
        return "\n".join(advice_lines)

print("✅ LangChainBPProcessor 클래스 정의 완료")


# # ============================================================
# # 4. 시스템 테스트 - 프로세서 초기화
# # ============================================================
# # LangChain 혈압 AI 분석 시스템 테스트
# print("🧠 LangChain 혈압 AI 분석 시스템 테스트")
# print("=" * 50)

# # 프로세서 초기화
# processor = LangChainBPProcessor()
# print("\n✅ 프로세서 초기화 완료")

# # ============================================================
# # 5. 테스트 1 - 개별 환자 AI 분석
# # ============================================================
# # 테스트용 환자 데이터
# test_patient = {
#     'age': 52,
#     'gender': '남성',
#     'bmi': 27.5,
#     'smoking': 1,
#     'exercise_frequency': 1,
#     'stress_level': 7,
#     'heart_rate_bpm': 82,
#     'family_history_hypertension': 1,
#     'systolic_bp': 145,
#     'diastolic_bp': 92
# }

# # 개별 환자 AI 분석
# print("\n👤 개별 환자 AI 분석:")
# analysis = processor.analyze_individual_bp(test_patient)

# print(f"\n분석 타입: {analysis['analysis_type']}")
# print(f"위험도: {analysis['risk_level']}")
# print(f"전반적 평가: {analysis['overall_assessment']}")
# print(f"\n주요 위험요인: {', '.join(analysis['key_risk_factors'])}")
# print(f"\n권장사항 ({len(analysis['recommendations'])}가지):")
# for i, rec in enumerate(analysis['recommendations'], 1):
#     print(f"  {i}. {rec}")
# print(f"\n생활습관 조언:\n{analysis['lifestyle_advice']}")
# print(f"\n추가 검진 필요: {'예' if analysis['follow_up_needed'] else '아니오'}")
# print(f"분석 출처: {analysis['source']}")

# # ============================================================
# # 6. 테스트 2 - 개인 맞춤 건강 조언 생성
# # ============================================================
# print("\n💡 개인 맞춤 건강 조언:")
# print("=" * 50)

# prediction_result = {
#     'systolic_bp': 145.0,
#     'diastolic_bp': 92.0
# }

# advice = processor.generate_health_advice(test_patient, prediction_result)
# print(advice)

# # ============================================================
# # 7. 테스트 3 - 데이터셋 AI 인사이트
# # ============================================================
# # 실제 데이터셋 로드
# try:
#     df = pd.read_csv('all_patient_features_preprocessed.csv')
#     print(f"✅ 데이터셋 로드 완료: {len(df)}명의 환자 데이터")
#     sample_df = df.sample(n=min(100, len(df)), random_state=42)
# except FileNotFoundError:
#     print("⚠️ 데이터셋 파일을 찾을 수 없습니다. 테스트 데이터로 대체합니다.")
#     sample_df = pd.DataFrame([test_patient] * 100)

# # 데이터셋 AI 인사이트
# print("\n📊 데이터셋 AI 인사이트:")
# print("=" * 50)

# dataset_analysis = processor.analyze_dataset_insights(sample_df)

# print(f"\n분석 타입: {dataset_analysis['analysis_type']}")
# print(f"분석된 환자 수: {dataset_analysis['total_patients']}")
# print(f"\n전체 요약:\n{dataset_analysis['summary']}")

# print(f"\n주요 패턴:")
# for i, pattern in enumerate(dataset_analysis['key_patterns'], 1):
#     print(f"  {i}. {pattern}")

# print(f"\n통계적 주요점:")
# for i, highlight in enumerate(dataset_analysis['statistical_highlights'], 1):
#     print(f"  {i}. {highlight}")

# print(f"\n임상적 의미:")
# for i, implication in enumerate(dataset_analysis['clinical_implications'], 1):
#     print(f"  {i}. {implication}")

# print(f"\n분석 출처: {dataset_analysis['source']}")