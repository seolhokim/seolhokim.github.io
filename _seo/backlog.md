# 콘텐츠 백로그 — 답할 질문의 대기열

> 운영 규칙은 `.claude/skills/fire-your-seo-agency/references/content.md` 4-1절.
> 우선순위: **노출 있고 순위 낮음 > 노출 있고 전용 페이지 없음 > 신규 추정 수요**.
> 아래 항목은 2026-10-20 기준 **검색 데이터 없이 글 목록에서 추정**한 시드다(최하 우선순위).
> GSC·네이버 서치어드바이저 등록 후 검색어 Top을 읽고 이 표를 실데이터로 교체한다.

## 상태 범례
backlog(대기) · draft(집필) · review(검수) · published(발행) · refresh-needed(갱신 필요) · merged(병합)

## 1. 기존 글 갱신 후보 (refresh) — 직답 첫 문단 + 기준일 + FAQ 추가

| 질문(검색창 표현) | 근거 | 유형 | 상태 | 대상 URL |
|---|---|---|---|---|
| CQL 오프라인 강화학습 설명 | 최신 글(2021-12), offline-rl 태그 허브 후보 | question | backlog | /reinforcement%20learning/2021/12/16/cql.html |
| Decision Transformer 설명 | 2021-10 글, 인용 가능성 높은 주제 | question | backlog | /reinforcement%20learning/2021/10/01/decision-transformer.html |
| DreamerV2 논문 리뷰 | Dreamer·PlaNet·DreamerV2 3편 → 모델 기반 RL 허브 | question | backlog | /reinforcement%20learning/2021/09/26/dreamerv2.html |
| deadly triad 강화학습 | 개념 정의형(용어 페이지로 재구성 시 추출 1순위) | glossary | backlog | /reinforcement%20learning/2021/09/02/deadly-triad.html |
| natural policy gradient 유도 | 수식 유도 글, 정의+표 구조로 보강 | glossary | backlog | /reinforcement%20learning/2021/01/12/natural-policy-gradient.html |
| on-policy off-policy 차이 | 정의형 질문, 비교 표 적합 | comparison | backlog | /reinforcement%20learning/2019/11/05/onpolicy-offpolicy.html |
| 강화학습 TSP attention 논문 | 조합최적화 7편(2021-07) → 허브 후보 | question | backlog | /reinforcement%20learning/2021/07/27/attention_tsp.html |
| hindsight experience replay 설명 | 2020-04 글 | question | backlog | /reinforcement%20learning/2020/04/03/hindsight-experience-replay.html |

## 2. 허브 페이지 후보 (hub) — 얇은 목록이 아니라 개요·표·직답이 있는 독립 페이지

| 허브 | 묶이는 글 | 상태 |
|---|---|---|
| 모델 기반 강화학습(Model-based RL) 정리 | world-model, PlaNet, Dreamer, DreamerV2, MuZero, model-based-RL | backlog |
| 오프라인 강화학습(Offline RL) 정리 | CQL, Decision Transformer, gMLP(offline-rl 태그) | backlog |
| 강화학습으로 푸는 조합최적화(TSP·VRP) | nco, s2v, attention_tsp, vrp, linear_tsp, alphago_zero_tsp, eco_dqn, learning_heuristics, combinatorial_rl | backlog |
| 탐험(Exploration) 기법 정리 | RND, curiosity, surprise, NGU, Agent57 | backlog |
| 선형대수 핵심 시리즈 안내 | 2020-12 선형대수 15편 | backlog |

## 3. 신규 질문 후보 (추정 수요 — 검색 데이터로 확인 전에는 쓰지 않는다)

| 질문 | 근거 | 유형 | 상태 |
|---|---|---|---|
| (GSC/서치어드바이저 등록 후 검색어 Top에서 채운다) | | | |

## 브리프 템플릿 (집필 전 5줄)

```
질문:       (검색창에 치는 문장 그대로)
직답:       (한 문장, 40자 내외, 수치·날짜 포함)
근거:       (표 1개로 만들 데이터 + 기준일 + 출처 URL)
하위 질문:  (FAQ가 될 3개)
내부 링크:  (허브 1 + 관련 글 2)
```
