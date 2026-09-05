# core/live_detector.py — 실시간 프레임 입력(폰 카메라 등)을 위한 보조 모듈
# detector.py, config.py는 건드리지 않고 별도 파일로 분리
import base64

import cv2
import numpy as np

from core.detector import Detector


def decode_base64_jpeg(data: str):
    """base64로 인코딩된 JPEG 문자열을 OpenCV 이미지(BGR ndarray)로 디코딩.
    'data:image/jpeg;base64,...' 형태의 data URL 접두사가 붙어 있어도 처리하고,
    디코딩에 실패하면 None을 반환한다 (예외를 던지지 않음)."""
    try:
        if data.strip().startswith("data:") and "," in data:
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data, validate=False)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # 디코딩 실패 시 cv2가 None을 반환
    except Exception:
        return None


class LiveDetector:
    """Detector를 감싸서 실시간 세션에 맞는 정리 로직을 추가한 래퍼.

    기존 /ws/detect는 영상 하나를 처음부터 끝까지 재생하고 끝나면 연결이 끊기지만,
    실시간 세션은 사람이 화면을 껐다 켰다 하듯 같은 track_id가 없어졌다 새로
    생기는 일이 훨씬 잦고 연결이 오래 지속된다. Detector.class_history는
    defaultdict라 한 번 생긴 track_id 항목이 저절로 사라지지 않으므로,
    이 래퍼가 '오래 안 보인 track_id'를 주기적으로 정리해 메모리가 계속
    쌓이는 것을 막는다.
    """

    def __init__(self, model_path, conf, history_size, stale_after_frames=90):
        self.detector = Detector(model_path=model_path, conf=conf, history_size=history_size)
        self.stale_after_frames = stale_after_frames  # 이만큼 연속으로 안 보이면 정리 대상
        self.frame_count = 0
        self.last_seen: dict[int, int] = {}  # track_id -> 마지막으로 보인 frame_count

    def detect(self, frame):
        self.frame_count += 1
        detections, annotated = self.detector.detect(frame)

        for d in detections:
            self.last_seen[d["track_id"]] = self.frame_count

        self._cleanup_stale()
        return detections, annotated

    def _cleanup_stale(self):
        """last_seen 기준으로 오래된 track_id를 last_seen과 class_history에서 제거"""
        stale_ids = [
            tid for tid, seen in self.last_seen.items()
            if self.frame_count - seen > self.stale_after_frames
        ]
        for tid in stale_ids:
            self.last_seen.pop(tid, None)
            self.detector.class_history.pop(tid, None)
