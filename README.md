# pladis-vla

VLA(GR00T N1.7, π0-base)의 DiT attention에 PLADIS-류 sparse 개입을
**query그룹(state/action) × key그룹(image/text) 셀별**로 적용하고,
LIBERO-plus perturbation 축(language/layout/robot/original)에서
개입 **locus**별 효과를 paired 검정으로 측정하는 실험 저장소.

이 문서는 제3자가 처음부터 끝까지 따라 실행할 수 있는 것을 목표로 한다.
순서: [환경 준비](#1-환경-준비) → [검증 게이트](#3-검증-게이트-스윕-전-필수) →
[실행](#4-실행) → [분석](#5-분석). 규약과 함정은 [6](#6-재현성결정론-규약)·[7](#7-함정-목록)절.

**설계 원칙 — 이 repo가 존재하는 이유:**
이전 반복(RLinf 위 패치)에서 난 버그들이 전부 "소유하지 않은 코드의 숨은 규약"에서
났다 (SDPA bool-마스크 규약 / Ray 메모리 모니터 / plus 모드가 지시문을 원본으로 전달 /
RLinf 서빙 래퍼 −35pp). 따라서 여기서는 **모델 로딩을 제외한 핵심 경로(환경 구동,
지시문 전달, 시드, 로깅)를 전부 이 repo 안의 코드가 소유**하고, 모든 주장
(전달됐다/같다/다르다)은 실물 데이터 검증 스크립트로 남긴다.

## 0. 구조

```
pladis/        방법론 훅
  attn_gr00t.py       GR00T N1.7 DiT용 (qgroup/kind 게이팅; weight-space 블렌드,
                      λ=0은 네이티브 SDPA 위임 = vanilla와 비트동일)
  attn_gr00t_fused.py [스테이징] fused-anchored 규약: SDPA + λ·(sparse−dense)@V,
                      λ 게이트 없음 — 교체 절차는 6절 참조
  attn_pi0.py         π0/π0.5 gemma용 (n_lang 자동 유도)
harness/       우리 소유의 최소 평가 루프 (Ray 없음)
  env.py              큐레이션 스케줄 + bddl/init 해석, 축별 전달 규약
  rollout.py          obs → policy → step 루프, 매 chunk 노이즈 핀, 영상 오버레이
  eplog.py            에피소드 단위 TSV 원장 (resume의 기준)
  model_gr00t.py      공식 Gr00tPolicy 어댑터 (RLinf 서빙 래퍼 −35pp 버그 대체)
  video.py            agentview+wrist mp4 녹화 (결정론 무접촉 검증됨)
experiments/   run.sh(환경 래퍼) + eval_arm.py(범용 러너)
               + sweep_n17_*.sh(스윕 드라이버) + verify_*.py(검증 게이트)
analysis/      analyze.py(--language|--layout) + analyze_robot.py — paired McNemar 집계
docs/          benchmark_facts.md (논문·코드 교차검증 사실 정리)
results/       (gitignore) 실행 산출물 — 로컬 전용, 레포에는 코드만
```

## 1. 환경 준비

### 1.1 하드웨어·소요

- CUDA GPU 1장 (bf16; 개발은 단일 H100), 에피소드당 ~13-17s
- 디스크: 체크포인트 ~30GB + 스윕당 영상 ~5-10GB

### 1.2 외부 체크아웃 (형제 디렉터리, 이 repo에 없음)

| 경로 | 내용 | 준비 방법 |
|---|---|---|
| `../RLinf/gr00t_n1d7/` | **실행 venv** (아래 1.3; RLinf 코드는 import하지 않음 — venv 호스트일 뿐) | uv로 생성 후 패키지 설치 |
| `../LIBERO-plus` | 벤치마크 체크아웃 (bddl/init/assets, `benchmark/task_classification.json` 큐레이션, `.magick` ImageMagick 빌드) | GitHub 체크아웃 + `pip install -e` |
| `../models/GR00T-N1.7-LIBERO/{libero_10,libero_goal,libero_object,libero_spatial}` | suite별 체크포인트 | `huggingface-cli download nvidia/GR00T-N1.7-LIBERO --local-dir ../models/GR00T-N1.7-LIBERO` |
| `~/.cache/huggingface` | Cosmos-Reason2-2B 백본 (첫 로드 시 자동 다운로드) | HF 토큰만 준비 |
| `~/.hf_user_token` | HF 토큰 평문 파일 — run.sh가 런타임에 읽음. **커밋 금지** | 직접 생성 |

### 1.3 실행 venv (`../RLinf/gr00t_n1d7`)

Python **3.11.14** (uv). 검증된 고정 버전 — 임의 업그레이드 금지:

| 패키지 | 버전 | 비고 |
|---|---|---|
| torch | 2.6.0 | + torchvision 0.21.0, flash_attn 2.7.4.post1 |
| robosuite | **1.4.1** | pip 설치본 유지 (editable 교체 금지) |
| diffusers | 0.35.1 | 훅이 AttnProcessor2_0의 line-for-line 포트 — 버전 바뀌면 3절 parity 재검증 |
| entmax | 1.3 | sparse branch 필수 |
| mujoco | 3.6.0 | MUJOCO_GL=egl (run.sh가 설정) |
| transformers | 4.57.3 | numpy 2.4.6, opencv 4.11 |
| gr00t | 0.1.0 (editable) | 공식 Isaac-GR00T 체크아웃을 `gr00t_n1d7/gr00t/`에 `pip install -e` |
| liberoplus | 0.1.0 (editable) | `../LIBERO-plus`를 `pip install -e` — bddl/init 경로가 패키지에서 자동 해석됨 |

## 2. 실행 규약 (모든 명령의 공통 전제)

**모든 파이썬 실행은 `experiments/run.sh` 래퍼를 경유한다.** 래퍼가 EGL 렌더링,
MagickWand 라이브러리 경로, PYTHONPATH, HF 토큰, 고정 인터프리터를 설정한다.
인라인 env 접두나 직접 python 호출로 우회하지 말 것 (7절 함정 ①②③).

```bash
cd pladis-vla
bash experiments/run.sh experiments/smoke_gr00t.py   # GPU 스모크 (모델 로드+2 스텝)
```

## 3. 검증 게이트 (스윕 전 필수)

새 환경/새 머신/의존성 변경 후에는 아래를 순서대로 통과시킨 뒤에만 스윕을 돌린다.

1. **앵커**: 표준 LIBERO-10 원본에서 모델카드 수치 재현 확인
   (`--axis none --episodes 100` → 91.0% @ n=100, 공식 94.35%와 표본오차 내).
2. **지시문 전달 스모크**: `--axis language` 소량 실행 후 eplog의 `instruction` 열이
   실제 perturbed 문장인지 확인 (RLinf 시절 버그 #3의 재발 방지).
3. **λ=0 parity**: `bash experiments/run.sh experiments/verify_base0_parity.py`
   — 훅 설치+λ=0 ≡ 미설치, 모듈 torch.equal + 2ep 롤아웃 eplog **비트 단위** 일치.
4. **축별 전달 게이트**: `verify_layout_axis.py`, `verify_robot_axis.py`
   — 교란 실재·결정론 bit-일치·장면 페어링·풀경로 모델 패리티.
5. (fused-anchored 규약 사용 시) `verify_fused_anchor.py cpu` → `cuda` — 6.3절 참조.

## 4. 실행

### 4.1 단일 arm — `eval_arm.py`

```bash
bash experiments/run.sh experiments/eval_arm.py \
  --suite libero_10 --axis language --episodes 0 --seed 0 \
  --model-path ../models/GR00T-N1.7-LIBERO/libero_10 \
  --out results/my_arm_eplog.tsv \
  [--video-dir results/videos/my_arm] \
  [--pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text]
```

| 플래그 | 의미 |
|---|---|
| `--suite` | libero_10 \| libero_goal \| libero_object \| libero_spatial (suite별 체크포인트 필수) |
| `--axis` | language \| layout \| robot \| … \| **none**(무교란 원본) |
| `--episodes` | **0 = 큐레이션 전수 1회씩** (seed-0 스케줄, arm 간 페어링); N>0 = 앞 N개 |
| `--seed` | 스케줄·env 리셋·flow 노이즈 시드의 단일 원천 |
| `--out` | eplog TSV. **resume 원장** — 이미 기록된 에피소드는 스킵 (재실행=이어달리기, 진짜 재실행은 파일 이동 후) |
| `--pladis-*` | 훅 설치는 명시 플래그로만 (환경변수 `PLADIS_ENABLE` 금지 — 로더가 crash) |

**Arm 조합표** (스윕 드라이버가 쓰는 그대로):

| arm | 플래그 |
|---|---|
| vanilla | (플래그 없음) |
| base0 | `--pladis-install --pladis-scale 0` (현 규약: SDPA 위임 = vanilla와 비트동일) |
| base0 (구 기준, eager-dense) | `--pladis-install --pladis-scale 1.0 --pladis-method softmax` |
| action×text 등 4셀 | `--pladis-install --pladis-scale 1.0 --pladis-qgroup {action,state} --pladis-kind {text,image}` |
| all×all | `--pladis-install --pladis-scale 1.0 --pladis-qgroup all --pladis-kind all` |

### 4.2 스윕 드라이버 — `sweep_n17_<axis>.sh`

```bash
nohup bash experiments/sweep_n17_robot.sh > results/sweep/driver_robot.out 2>&1 &
```

| 드라이버 | 내용 | 규모 |
|---|---|---|
| `sweep_n17_language.sh` | 7 arms × 1,537 eps × 4 suites | ~31h |
| `sweep_n17_original.sh` | 7 arms × 400 eps (태스크×init 0-9) — 기준선 | ~9h |
| `sweep_n17_layout.sh` | 7 arms × 1,525 eps | ~31-35h |
| `sweep_n17_robot.sh` | 7 arms × 1,550 eps — λ arms 후 구 기준 base0 arm이 parity 게이트(2ep language vs 저장된 구 base0 eplog) 통과 시 이어짐. λ=0-as-base0은 vanilla와 비트동일이라 없음 | ~48h |

전부 에피소드 단위 resume-safe. 일부 드라이버는 선행 작업 감시(polling) 후 기동하는
큐잉 로직을 내장 — 파일 상단 주석 참조. 산출물 명명: `results/sweep/n17_{axis}_{arm}_{suite}_eplog.tsv`
+ 같은 이름 `.out` 로그 + `videos/n17_{axis}_{arm}_{suite}/ep#####_{S|F}_{task}.mp4`.

eplog 스키마(TSV): `episode, task_name, base_task, init_state_id, instruction,
success_once, success_at_end, n_steps, wall_s`.

### 4.3 프로세스 확인

`pgrep -f`는 bash -c 래퍼를 잡는다. 드라이버 pid는:
```bash
ps -eo pid,etime,args | grep "[s]weep_n17"
```

## 5. 분석

```bash
python3 analysis/analyze.py --language    # n17_lang_*  (locus 대비 + vs 두 기준선)
python3 analysis/analyze.py --layout     # n17_layout_*
python3 analysis/analyze_robot.py        # n17_robot_* (level 0.1-0.5 용량-반응 포함)
```

판정 규약: **paired McNemar** (z = (n01−n10)/√discordant, 같은 seed-0 스케줄로
에피소드 페어링), pooled 우선. 단일-suite 단일대비는 |z|≲2.7 요동이 순수 수치경로
노이즈만으로 실측된 바 있으므로 할인한다. 다중비교는 Bonferroni 명시.

## 6. 재현성·결정론 규약

### 6.1 결정론의 3중 고정

1. **스케줄**: `env.py`의 로컬 `np.random.default_rng(seed)` 순열 — 같은 (suite, axis, n, seed)면 항상 같은 에피소드 리스트 (arm 간 페어링의 근거).
2. **env 리셋**: 매 리셋 직전 `env.seed(seed·1_000_003 + episode)` — BDDL 배치 샘플링이 에피소드 번호에만 종속.
3. **flow 초기 노이즈**: 매 chunk 추론 직전 `torch.manual_seed(episode_seed·100_003 + step)` (`rollout.py`) — arm 간 동일 노이즈 스트림.

⇒ 같은 머신·같은 버전에서 **비트 단위 재현** (vanilla×2 완전일치 실증).
다른 GPU/커널 버전에서는 비트동일이 깨질 수 있으나 통계적 결론은 유지 (아래 6.2).

### 6.2 수치경로 규약 (중요)

- 현 훅(`attn_gr00t.py`)은 **weight-space** 블렌드: λ>0이면 수동(eager) 경로에서
  `dense + λ(sparse−dense)`를 만들어 @V. 원저 PLADIS 공개코드와 동일 구성.
- vanilla(fused SDPA)와 λ>0(eager)의 **커널 전환 항**은 4축 누적 실측 −0.80pp
  (z=−1.95, n=5,012 paired) — 결론을 바꾸지 않는 크기지만, vs-vanilla 단일대비
  해석 시 이 항의 존재를 명시할 것. 커널-매칭 대조가 필요하면 구 기준 base0
  arm(4.1의 `--pladis-method softmax` 행)을 기준선으로 쓴다.
- **fused-anchored 규약 (스테이징, `attn_gr00t_fused.py`)**: 선형 분해
  `(d+λ(s−d))@V = SDPA + λ(s−d)@V`로 dense 앵커를 vanilla와 동일한 fused 호출로
  공유, λ=0에서도 보정항을 생략 없이 계산(+0·corr) → base0≡vanilla가 구성상 성립
  하면서 검증력을 가짐. **교체 절차**: 진행 중인 스윕이 없는지 확인 →
  `cp pladis/attn_gr00t_fused.py pladis/attn_gr00t.py` →
  `verify_fused_anchor.py cuda` 통과 → 2ep 롤아웃 parity. 두 규약의 λ>0 출력 차는
  dtype 라운딩 바닥(bf16 ~4e-3 상대)으로 동일 방법임이 검증되어 있음.

## 7. 함정 목록 (전부 실사고)

1. **import 순서**: MagickWand dlopen은 torch/cv2 로드 후 실패 — liberoplus.envs를
   모델보다 먼저 import (`model_gr00t.py`가 강제; 직접 스크립트 작성 시 주의).
2. **환경변수 훅 금지**: `PLADIS_ENABLE` 설정 시 로더 crash. 훅은 `install_pladis()` 명시 호출만.
3. **run.sh 우회 금지**: 인라인 env 접두 실행은 EGL/MagickWand/인터프리터가 어긋난다.
4. **bool 마스크**: SDPA는 bool을 False=−inf로 해석. `logits+mask` 덧셈은 마스킹을
   조용히 무력화 (실제 버그였음). 훅의 additive 변환 경로를 거칠 것.
5. **지시문 전달**: 벤치마크 래퍼가 variant를 심에만 로드하고 모델엔 원본 지시문을
   주는 사고가 있었음 — eplog `instruction` 열로 항상 실측 확인.
6. **eplog resume**: 같은 `--out`으로 재실행하면 전부 스킵된다. 의도된 동작.
7. **절대경로**: 세션/스크립트에서 cwd가 리셋되는 사고 다발 — nohup 서브셸 포함
   항상 절대경로 또는 명시적 `cd`.
8. **랜덤-텐서 단위테스트 불신**: 균일 마스크는 softmax-불변이라 가짜 통과함 —
   poisoned-key·실물 롤아웃 검증으로만 판정 (verify_* 스크립트들이 그 산물).

## 8. 완료된 스윕 (2026-07 기준)

| 축 | 완료 | 핵심 결과 (pooled) |
|---|---|---|
| language | 07-16 | a×t−a×i = +3.77pp z=+4.28 (locus 분리); a×i 유해 −3.19 z=−4.0 |
| original | 07-17 | per-task 기준선 (n=10/task) |
| layout | 07-19 | 전면 NULL (locus +0.26 z=+0.31) — 이득의 grounding-특이성 |
| robot | 07-22 | locus +3.10 z=+2.93 (a×i 병변 −3.35 z=−3.33 기인, a×t 무해·무익) |

종합: **text-cross sharpening은 전 축 무해(이득은 지시문-OOD에서만), image-cross
해악은 2/3축 재현** — 개입 locus가 결과를 가른다. 수치·검정은 analysis/ 스크립트로 재생성.
