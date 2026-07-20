# pladis-vla

VLA(GR00T N1.7, π0-base)의 DiT attention에 PLADIS-류 sparse 개입을
query그룹(state/action) × key그룹(image/text) 셀별로 적용하고,
LIBERO-plus perturbation 축에서 효과를 측정하는 실험 저장소.

**설계 원칙 — 이 repo가 존재하는 이유:**
이전 반복(RLinf 위 패치)에서 난 버그 3건이 전부 "소유하지 않은 코드의 숨은 규약"에서
났다 (SDPA bool-마스크 규약 / Ray 메모리 모니터 / plus 모드가 지시문을 원본으로 전달).
따라서 여기서는 **모델 로딩을 제외한 핵심 경로(환경 구동, 지시문 전달, 시드, 로깅)를
전부 이 repo 안의 코드가 소유**한다. 모든 주장(전달됐다/같다/다르다)은 실물 데이터
검증을 코드로 남긴다 — 랜덤-텐서 단위테스트는 신뢰하지 않는다(bool-마스크 버그가 통과했음).

## 구조

```
pladis/        방법론 훅 (검증 완료본을 RLinf 패치에서 이전)
  attn_gr00t.py   GR00T N1.7 DiT용 (qgroup/kind 게이팅; λ=0은 네이티브 SDPA 위임
                  = vanilla와 비트 동일, 원저 PLADIS의 λ=0 게이트 의미론)
  attn_pi0.py     π0/π0.5 gemma용 (n_lang 자동 유도 포함)
harness/       우리 소유의 최소 평가 루프 (Ray 없음)
  env.py          큐레이션 스케줄 + bddl/init 해석, 축별 전달 규약
                  (language/light/background=init 적용, layout=scene-altering
                  reseed, robot=런타임 initstate → 공식 프로토콜 그대로)
  rollout.py      obs → policy → step 루프, 매 chunk 노이즈 핀, 영상 오버레이
  eplog.py        에피소드 단위 TSV 원장 (resume의 기준)
  model_gr00t.py  공식 Gr00tPolicy 어댑터 (RLinf 서빙 래퍼 -35pp 버그 대체)
  video.py        agentview+wrist mp4 녹화 (결정론 무접촉 검증됨)
experiments/   run.sh(env 래퍼) + eval_arm.py(범용 러너) + sweep_n17_*.sh(드라이버)
               + verify_*.py(게이트: base0 비트동일 / layout / robot 축)
analysis/      집계·paired 검정(McNemar)·그림
results/       (gitignore) 실행 산출물 — 로컬 전용
```

## 검증 게이트 (순서 고정)

1. **앵커**: 표준 LIBERO-10에서 GR00T N1.7 모델카드 수치 재현 — 하네스 자체 검증
   (공식 Gr00tPolicy 경로에서 91% @ n=100, 공식 94.35%와 표본오차 내).
2. **지시문 전달 스모크**: `_language` 변형 로드 시 모델 입력 문자열이 실제로
   perturbed 문장인지 eplog로 확인.
3. **λ=0 parity**: 훅 설치+λ=0 ≡ 미설치, **비트 단위**(`verify_base0_parity.py`
   모듈 torch.equal + 롤아웃 eplog 완전일치). λ=0이 네이티브 SDPA로 위임되므로
   성립 — 원저 PLADIS도 λ=0이면 프로세서 교체 자체를 안 함.
4. **축별 전달 게이트**: 교란이 실제로 모델에 도달하는지 실측
   (`verify_layout_axis.py`, `verify_robot_axis.py` — 결정론 bit-일치·교란 실재·
   장면 페어링·풀경로 모델 패리티).
5. 그 후에만 스윕 (`experiments/sweep_n17_<axis>.sh`).

## 외부 의존 (이 repo에 없는 것)

| 경로 | 내용 |
|---|---|
| `../RLinf` | 모델 로딩 코드(gr00t_n1d7/openpi 래퍼) + venv 2개(`gr00t_n1d7/`, `openpi/`) |
| `../LIBERO-plus` | 벤치마크 체크아웃 (bddl/init/assets, `.magick` ImageMagick) |
| `../models` | 체크포인트: GR00T-N1.7-LIBERO, RLinf-Pi0-LIBERO-Long-SFT |
| `~/.cache/huggingface` | Cosmos-Reason2-2B 백본 |
| `/home/reallab/.hf_user_token` | HF 토큰 (런타임 읽기, 커밋 금지) |

## 이전 반복의 교훈 (요약)

- **bool 마스크**: SDPA는 bool을 False=−inf로 해석. `logits+mask` 덧셈은 마스킹을
  조용히 무력화. 균일 마스크는 softmax-불변이라 parity가 통과해버림 → poisoned-key로 검증.
- **지시문 전달**: RLinf plus 모드는 variant bddl을 심에만 로드, 모델 지시문은 원본
  `task.language` → `_language` 축이 모델에 도달한 적 없음. 전달 경로는 반드시 직접 소유.
- **비결정성**: 동일 seed·동일 코드 2회 실행에서 에피소드 15.6% 뒤집힘 실측
  (flow init noise 미고정 + 폐루프 카오스). n=192에서 2SE≈7pp. 판정은 paired 검정
  + 용량-반응 + 다중 seed로만.
- LIBERO-plus 수치: Language Instructions = 1,537 (spatial 390/object 354/goal 410/long 383),
  bddl 파일로는 suite당 500 (10태스크×50변형).

## 데이터 흐름도

(harness 작성과 함께 채운다 — 지시문/이미지/상태/노이즈 각각의 출처와 소비처를
file:line으로 명시하는 것이 이 문서의 완성 조건)
