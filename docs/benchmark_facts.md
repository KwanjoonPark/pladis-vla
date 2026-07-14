# LIBERO-plus 벤치마크 사실 (논문 + 코드 교차검증, 2026-07-14)

출처: 논문 PDF(workspace 루트, arXiv:2510.13626v3) + `../LIBERO-plus` 체크아웃 코드.
모든 항목은 두 출처가 일치함을 확인한 것만 기록.

## 규모 (Appendix C, Table 7)

- 총 10,030 태스크 = 7축 × 21 하위성분. 생성은 suite×축당 500개(총 14,000) →
  전 모델이 푸는 쉬운 태스크 제거(ceiling filtering) → 10,030 큐레이션.
- **Language Instructions = 1,537**: Spatial 354 / Object 390 / Goal 410 / **Long(libero_10) 383**
  (paper Table 7; 동봉 task_classification.json은 spatial↔object 수치가 서로 바뀌어 있으나
  goal/long/총합은 일치 — libero_10=383은 양쪽 확정).
- 디스크 bddl은 suite당 `_language` 500개(10 base × 50 변형) = **큐레이션 전 전체 생성본**.
  ⚠️ 논문 비교가 목적이면 task_classification.json의 383개만 돌릴 것.
- 언어 축 하위 3종 (Appendix A.6): R1 Distraction(길고 산만한 문장),
  R2 Common Sense(물체 명칭을 상식적 서술로 치환), R3 Reasoning Chain(추론 재구성).

## 축별 구현 방식 (Appendix A + env_wrapper.py)

| 축 | 구현 | bddl 내용 변화 |
|---|---|---|
| Language | bddl `(:language)` 줄만 rewrite | 장면 동일 (diff로 확인: :language 외 0 diff) |
| Light / Background | scene XML 수정, `_light`/`_table`/`_tb` bddl | 있음 |
| Layout | `_add`(교란물체)/`_level`/`_sample` bddl | 있음 (⚠️ 상태벡터 차원 변동) |
| Camera | **런타임**: 파일명 `_view_<h>_<v>_<scale>_<rot>_<vert>` 파싱 | 없음 |
| Robot Init | **런타임**: 파일명 `_initstate_<k>` → 로봇 클래스명 치환(초기 qpos 교란 0.1–0.5) | 없음 |
| Sensor Noise | **런타임**: 파일명 `_noise_<n>` → 이미지 photometric 변형 (Table 5: motion/gaussian/zoom blur, fog, glass) | 없음 |

- **기본값 = 전부 OFF**: `env_wrapper.py:204` — 파일명에 `_view_`+`_initstate_` 마커가
  없으면 camera/robot/noise 파라미터 전부 0/1.0 (219-226행). 순수 bddl 경로 로드 시
  숨은 randomizer 없음 → 축 격리 보장.

## 지시문의 공식 경로

- `env_wrapper.py:244`: `self.language_instruction = problem_info["language_instruction"]`
  (BDDLUtils.get_problem_info가 bddl에서 파싱). **harness는 이걸 모델에 전달**하고,
  에피소드별 eplog에 문자열을 기록해 전달을 데이터로 증명한다.
- (반면 RLinf plus 모드는 원본 `task.language`를 전달 — 언어 축 무효화 버그의 원인.)

## 초기상태 파일

- `init_files/<suite>/<base_task>.init` = (100, 47) float64, `.pruned_init` = (50, 47).
  torch.load 포맷. base 태스크당 1쌍(변형별 아님). 표준 평가 = pruned 50.
- `_language`(및 light/background)는 장면 동일 → base의 pruned_init 50 그대로 적용 가능
  (dim-일치 assert 필수). `_add` 계열은 차원 불일치로 불가.
- 논문의 "Robot Initial States" 축은 이 파일과 무관 (위 표의 런타임 qpos 교란).

## 평가 프로토콜 참고 (paper)

- 카메라 규격: agentview + robot0_eye_in_hand (env_wrapper 기본 128×128 — 모델 요구에 맞게 지정).
- 난이도 L1–L5: 4개 대표 모델 중 몇 개가 풀었는지로 층화 (Appendix C.3).
- π0의 LIBERO-plus 전체 평균 53.6, Language 축 58.8 (Table 2) — π0 트랙 앵커로 사용 가능.
- 논문 발견: 모델들은 언어를 사실상 무시(Finding 3/7/8; blank-instruction 실험) —
  우리 text-게이팅 개입의 동기와 직결. blank-instruction 대조군은 우리 실험에도 저렴한 진단.
