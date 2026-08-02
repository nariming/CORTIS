"""
Agentic 순환 로직.

C(이벤트 감지+예측)가 매번 새 이벤트를 확정할 때마다:
  1. 유저 히스토리에 이벤트 추가
  2. CohortIndex로 재검색
  3. Reasoner로 재예측 (근거 포함)
  4. [신규] A(대출상품 매칭) 재호출 -> 포트폴리오 결정 레이어로 "타겟 포트폴리오 + 결론 문단" 도출
  5. 예측 결과를 A(정책 재탐색)에 다시 넘김 (여기선 스텁 콜백)

"한 번 쓰고 끝나는 챗봇"이 아니라 "이벤트가 생길 때마다 스스로 다시 판단하는 에이전트"라는
지점을 코드 구조로 직접 보여주는 게 목적. A/B는 아직 개발자1 쪽 API가 없으므로 스텁으로 대체.
"""

from typing import Callable, List, Optional
from dataclasses import dataclass, field

from .similarity import CohortIndex
from .reasoning import Reasoner, PredictionResult
from .scenario_tree import build_scenario_tree
from . import backend_client
from .portfolio import (
    UserFinancialProfile, ExistingLoan, select_best_portfolio, build_watch_conditions,
    compute_event_based_timeline,
)
from .portfolio_summary import build_summary_input, get_summarizer, validate_summary_numbers


@dataclass
class AgentState:
    user_id: str
    confirmed_history: List[str]
    # State/Transaction Embedding 검색용. 미리 채워서 넘겨도 되고, None으로 두면
    # on_event_confirmed()가 backend_client.fetch_user_detail()/fetch_user_transactions()로
    # 자동으로 채운다(서버 미기동 등으로 실패하면 History만으로 안전하게 폴백).
    query_state: Optional[dict] = None
    query_tx: Optional[dict] = None


@dataclass
class PortfolioDecisionResult:
    """포트폴리오 결정 레이어의 최종 산출물. on_predict_callback에 함께 전달된다."""
    summary: dict                      # when/how_much/effect/comparison/next_action 5항목
    watch_conditions: List[str]
    dsr_before: float
    dsr_after: float
    npv_savings: int
    validated: bool                    # 검증가드 통과 여부


class CortisAgent:
    """오프라인/목업 환경에서 파이프라인 배선을 빠르게 검증하기 위한 에이전트.

    실제 서비스 흐름(이벤트 확정 -> 예측 저장 -> A파트 정책매칭)은
    demo_full_flow.py + pipeline/backend_client.py 가 실제 API로 수행한다.
    이 클래스는 서버 없이도 로직만 빠르게 테스트하고 싶을 때 쓰는 용도로 남겨둔다.
    """

    def __init__(
        self,
        index: CohortIndex,
        reasoner: Reasoner,
        on_predict_callback: Optional[Callable[[str, PredictionResult, List[str]], None]] = None,
        on_portfolio_callback: Optional[Callable[[str, PortfolioDecisionResult], None]] = None,
        top_k: int = 5,
        base_url: str = backend_client.BACKEND_BASE_URL,
    ):
        self.index = index
        self.reasoner = reasoner
        self.top_k = top_k
        self.base_url = base_url
        self.on_predict_callback = on_predict_callback or self._default_stub_callback
        self.on_portfolio_callback = on_portfolio_callback or self._default_portfolio_stub_callback

    def _default_stub_callback(self, user_id: str, result: PredictionResult, updated_history: List[str]):
        """서버 없이 로직만 확인할 때 쓰는 콘솔 출력.

        라벨을 "[AGENT]"로 표기한다 — 이전엔 "[MOCK]"으로 고정 표기했는데, 이 콜백은
        MockReasoner든 AnthropicReasoner든 상관없이 항상 호출되는 공용 콜백이라
        실제로 LLM_BACKEND=anthropic으로 실행돼도 "[MOCK]"이 찍혀 오해를 부를 수 있었다.
        실제 A파트 호출은 demo_full_flow.py 의 backend_client.request_policy_match() 를 참고."""
        print(f"  [AGENT] 예측 완료 (user={user_id}, history={updated_history})")
        for pred in result.predictions:
            print(f"  [AGENT] -> A파트라면 '{pred['event']}' 관련 정책 매칭을 요청했을 것: {pred.get('reasoning', '')}")
        for s in result.scenarios:
            print(f"  [AGENT] {s.label}: {s.path_str()} (결합확률 {s.joint_probability_pct}%)")

    def _default_portfolio_stub_callback(self, user_id: str, result: PortfolioDecisionResult):
        print(f"  [AGENT] 포트폴리오 결정 완료 (user={user_id}, 검증통과={result.validated})")
        print(f"  [AGENT] {result.summary}")
        print(f"  [AGENT] 감시조건: {result.watch_conditions}")

    def on_event_confirmed(self, state: AgentState, new_event: str, user_context: Optional[str] = None) -> PredictionResult:
        """이벤트가 규칙기반 감지+확인질문을 거쳐 '확정'되었을 때 호출되는 진입점.

        (규칙기반 감지 자체는 A파트 모듈의 detect_* 함수들이 담당 — 여기선 이미
        확정된 이벤트가 들어온다고 가정)
        """
        state.confirmed_history.append(new_event)

        # State/Transaction Embedding 검색을 위해 필요한 값이 아직 안 채워졌으면 HTTP로 채운다.
        # (서버 미기동 등으로 실패하면 History만으로 안전하게 폴백 — CohortIndex.search()가
        # query_state/query_tx=None을 이미 그렇게 처리하도록 설계돼 있음)
        if state.query_state is None or state.query_tx is None:
            try:
                user_detail = backend_client.fetch_user_detail(state.user_id, self.base_url)
                txs = backend_client.fetch_user_transactions(state.user_id, self.base_url)
                state.query_state, state.query_tx = backend_client.build_query_state_and_tx(user_detail, txs)
            except Exception as e:
                print(f"  [WARN] State/Transaction 조회 실패 - History만으로 검색: {e}")

        matches = self.index.search(
            state.confirmed_history,
            query_state=state.query_state,
            query_tx=state.query_tx,
            top_k=self.top_k,
        )
        counts = self.index.aggregate_next_events(matches)
        result = self.reasoner.predict(
            confirmed_history=state.confirmed_history,
            matches=matches,
            next_event_counts=counts,
            user_context=user_context,
        )

        # Scenario Tree는 LLM을 호출하지 않는 별도 계산(코호트 재검색)이라 Reasoner 책임 밖에서
        # 채운다 — LLM 호출 실패/느림과 무관하게 항상 채워지도록 별도로 실행.
        try:
            result.scenarios = build_scenario_tree(
                self.index, state.confirmed_history,
                query_state=state.query_state, query_tx=state.query_tx, top_k=self.top_k,
            )
        except Exception as e:
            print(f"  [WARN] Scenario Tree 계산 스킵: {e}")

        self.on_predict_callback(state.user_id, result, state.confirmed_history)

        # [신규] 예측이 나온 즉시 포트폴리오 결정 레이어까지 이어서 실행
        # (실제 서버 필요 - 서버 없이 테스트하려면 이 호출 부분만 주석 처리하고 사용)
        try:
            self._run_portfolio_decision(state, new_event, result, event_confirmed=True)
        except Exception as e:
            print(f"  [WARN] 포트폴리오 결정 단계 스킵 (서버 미기동 등): {e}")

        return result

    @staticmethod
    def _top_prediction_timeline(prediction: PredictionResult):
        """예측된 '다음 이벤트' 후보 중 최상위(모순 없는 것 우선 정렬된 predictions[0])를
        EligibilityTimelineEntry로 변환한다 — 6단계에서 확률/시점/필요자금까지 구조화된
        예측 결과가 나왔으니, 이걸 이벤트 이름 하나로 뭉개지 않고 그대로 포트폴리오
        레이어(build_watch_conditions)에 넘기는 지점이 바로 여기다.

        predictions가 비어있으면(검색 결과 없음 등) None을 반환해 감시조건에서 자연스럽게
        제외되게 한다 — 근거 없이 이 항목을 채우지 않는다.
        """
        if not prediction.predictions:
            return None
        top = prediction.predictions[0]
        return compute_event_based_timeline(
            predicted_event=top["event"],
            evidence_count=top["evidence_count"],
            policy_or_product_name=top["event"],
            expected_timing_months=top.get("expected_timing_months"),
            expected_cash_need_krw=top.get("expected_cash_need_krw"),
            cash_need_source=top.get("cash_need_source"),
        )

    def _run_portfolio_decision(
        self,
        state: AgentState,
        event_name: str,
        prediction: PredictionResult,
        event_confirmed: bool,
    ):
        """새 이벤트가 확정/예측될 때마다 A(대출매칭)를 다시 불러 포트폴리오를 재계산.

        이게 "C가 감지할 때마다 A가 다시 도는 지속적 순환 구조"를 실제로 증명하는 부분.
        """
        user_detail = backend_client.fetch_user_detail(state.user_id, self.base_url)
        profile = backend_client.build_profile_from_user_detail(user_detail)
        loans = backend_client.build_existing_loans_from_user_detail(user_detail)
        if not loans:
            return  # 보유 대출 없으면 포트폴리오 재설계 대상 없음

        loan_match_response = backend_client.request_loan_match(
            state.user_id, event_types=[event_name], base_url=self.base_url,
        )
        loan_refinance_map = backend_client.build_refinance_map_from_loan_match(loans, loan_match_response)

        # [신규] 정책형 대출(버팀목전세자금대출 등)도 대환 후보로 함께 조회 - 놓치지 않기 위함
        try:
            policy_match_response = backend_client.request_policy_match(
                user_id=state.user_id, prediction_id=None, event_types=[event_name],
                persist=False, base_url=self.base_url,
            )
            policy_catalog = backend_client.fetch_policy_catalog(self.base_url)
            policy_refinance_map = backend_client.build_refinance_map_from_policy_match(
                loans, policy_match_response, policy_catalog,
            )
        except Exception as e:
            print(f"  [WARN] 정책형 대출 후보 조회 실패 (KB 자체상품만으로 진행): {e}")
            policy_refinance_map = {}

        refinance_map = backend_client.merge_refinance_maps(loan_refinance_map, policy_refinance_map)

        best, second, infeasible = select_best_portfolio(loans, refinance_map, profile)
        if best is None:
            print(f"  [WARN] 실행가능한 포트폴리오 없음: {infeasible}")
            return

        dsr_before = round((sum(l.monthly_payment for l in loans) * 12 / profile.annual_income) * 100, 1)

        evidence_count = next(
            (p.get("evidence_count", 0) for p in prediction.predictions if p.get("event") == event_name), 0
        )
        summary_input = build_summary_input(
            user_name=user_detail.get("name", state.user_id),
            event_name=event_name,
            loans=loans,
            profile=profile,
            best=best,
            second=second,
            dsr_before=dsr_before,
            next_action_hint="정확한 신청 방법·필요서류는 해당 상품 안내 페이지에서 최종 확인 필요",
        )

        summarizer = get_summarizer()
        summary = summarizer.summarize(summary_input)
        validated, bad_numbers = validate_summary_numbers(summary, summary_input.allowed_numbers())
        if not validated:
            print(f"  [WARN] LLM 출력에 검증 안 된 숫자 발견: {bad_numbers} (사용자에게 노출 전 재검토 필요)")

        watch_conditions = build_watch_conditions(
            best, event_name, event_confirmed,
            predicted_next_event=self._top_prediction_timeline(prediction),
        )

        decision = PortfolioDecisionResult(
            summary=summary,
            watch_conditions=watch_conditions,
            dsr_before=dsr_before,
            dsr_after=best.dsr_pct,
            npv_savings=best.npv_savings_vs_keep,
            validated=validated,
        )
        self.on_portfolio_callback(state.user_id, decision)