# SPDX-License-Identifier: Apache-2.0
"""Per-episode mp4 recording of exactly what the policy saw, with a debug
overlay: the instruction plus a model/arm tag (e.g. "GR00T-N1.7-LIBERO/libero_10
| action x text (scale=1)") in a header bar, a step counter badge, and a 1s
SUCCESS/FAIL end card (streaming encode cannot retro-label earlier frames).
Overlay text must stay ASCII — cv2 Hershey fonts render nothing else.

Frames are the two camera streams fed to the model (agentview + wrist, both
180°-rotated as in rollout.wrap_obs_gr00t), hstacked to (H, 2W, 3). The
recorder is a pure observation consumer — it never touches torch RNG or the
model path — so recorded and unrecorded runs are step-identical.

Files are written as ep#####_TMP_<task>.mp4 and renamed on close to _S_
(success_once) or _F_; a leftover TMP file marks a crashed episode.
"""

from __future__ import annotations

import os

import cv2
import imageio.v2 as imageio
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SCALE = 0.42


def _wrap_px(text: str, max_w: int, max_lines: int = 3) -> list:
    """Greedy word wrap by measured pixel width (not char count)."""
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if not cur or cv2.getTextSize(cand, _FONT, _SCALE, 1)[0][0] <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += " ..."
    return lines


class EpisodeVideo:
    """One mp4 per episode: add() once after reset and per control step,
    then close(success)."""

    def __init__(
        self,
        video_dir: str,
        episode: int,
        task_name: str,
        instruction: str,
        label: str = "",
        fps: int = 20,
    ):
        os.makedirs(video_dir, exist_ok=True)
        self._dir = video_dir
        self._episode = episode
        self._task = task_name
        self._fps = fps
        self._instruction = instruction
        self._label = label
        # wrapping needs the frame width -> laid out lazily on the first add()
        self._lines = None
        self._header_h = None
        self._tmp = os.path.join(video_dir, f"ep{episode:05d}_TMP_{task_name}.mp4")
        self._writer = imageio.get_writer(
            self._tmp, fps=fps, codec="libx264", quality=6, pixelformat="yuv420p"
        )
        self._last = None
        self.n_frames = 0

    def _compose(self, raw_obs: dict) -> np.ndarray:
        main = raw_obs["agentview_image"][::-1, ::-1]
        wrist = raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]
        sim = np.ascontiguousarray(np.hstack([main, wrist]))
        if self._lines is None:
            self._lines = _wrap_px(self._instruction, sim.shape[1] - 12)
            # the step badge sits right-aligned on the label row; keep the
            # label clear of the widest badge we expect ("step 9999")
            badge_w = cv2.getTextSize("step 9999", _FONT, 0.45, 1)[0][0]
            if self._label:
                self._label = _wrap_px(
                    self._label, sim.shape[1] - badge_w - 24, max_lines=1
                )[0]
            # round total height up to a multiple of 16 (ffmpeg macro block) via
            # the header, so the writer never resizes (blurs) the frame
            n_rows = len(self._lines) + 1  # +1: label row
            self._header_h = -(-(18 * n_rows + 22) // 16) * 16
        header = np.full((self._header_h, sim.shape[1], 3), 24, np.uint8)
        for i, line in enumerate(self._lines):
            cv2.putText(header, line, (6, 16 + 18 * i), _FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        if self._label:
            cv2.putText(header, self._label, (6, self._header_h - 7),
                        _FONT, 0.42, (120, 190, 250), 1, cv2.LINE_AA)
        badge = f"step {self.n_frames}"
        bx = sim.shape[1] - cv2.getTextSize(badge, _FONT, 0.45, 1)[0][0] - 6
        cv2.putText(header, badge, (bx, self._header_h - 7),
                    _FONT, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        return np.vstack([header, sim])

    def add(self, raw_obs: dict) -> None:
        frame = self._compose(raw_obs)
        self._writer.append_data(frame)
        self._last = frame
        self.n_frames += 1

    def close(self, success: bool) -> str:
        if self._last is not None:  # 1s end card: verdict burned over the last frame
            card = self._last.copy()
            tint = np.zeros_like(card)
            tint[:] = (30, 120, 30) if success else (120, 30, 30)
            card = cv2.addWeighted(card, 0.65, tint, 0.35, 0)
            text = "SUCCESS" if success else "FAIL"
            size = cv2.getTextSize(text, _FONT, 1.6, 3)[0]
            org = ((card.shape[1] - size[0]) // 2, (card.shape[0] + size[1]) // 2)
            cv2.putText(card, text, org, _FONT, 1.6,
                        (120, 255, 120) if success else (255, 120, 120), 3, cv2.LINE_AA)
            for _ in range(self._fps):
                self._writer.append_data(card)
        self._writer.close()
        final = os.path.join(
            self._dir,
            f"ep{self._episode:05d}_{'S' if success else 'F'}_{self._task}.mp4",
        )
        os.replace(self._tmp, final)
        return final
