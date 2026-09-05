# benchmark_detector.py — LiveDetector(=Detector 재사용)의 순수 추론 처리량을 측정
# data/30s.mp4에서 프레임을 뽑아 detect()만 반복 호출해 프레임당 소요 시간을 잰다
# (네트워크/디코딩 오버헤드를 뺀 "서버가 실제로 처리 가능한 FPS"를 보기 위함)
import time

import cv2

import config
from core.live_detector import LiveDetector

VIDEO_PATH = "data/30s.mp4"


def run_benchmark(resize_to=None, max_frames=120):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[오류] 영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    live_detector = LiveDetector(
        model_path=config.MODEL_PATH,
        conf=config.CONFIDENCE,
        history_size=config.HISTORY_SIZE,
    )

    frame_times = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_to:
            frame = cv2.resize(frame, resize_to)

        t0 = time.perf_counter()
        live_detector.detect(frame)
        frame_times.append(time.perf_counter() - t0)
        count += 1

    cap.release()

    if not frame_times:
        print("[경고] 처리된 프레임이 없습니다")
        return

    # 첫 프레임은 모델 warmup(그래프 컴파일 등)이 섞여 왜곡되므로 통계에서 제외
    warmup, rest = frame_times[0], frame_times[1:]
    avg = sum(rest) / len(rest) if rest else warmup
    fps = 1 / avg if avg > 0 else float("inf")

    label = f"{resize_to[0]}x{resize_to[1]}" if resize_to else f"원본({native_w}x{native_h})"
    print(f"[{label}] 프레임 {len(frame_times)}개 | 첫 프레임(warmup) {warmup*1000:.1f}ms | "
          f"평균 {avg*1000:.1f}ms/frame | 추정 FPS {fps:.2f}")


if __name__ == "__main__":
    run_benchmark(resize_to=None, max_frames=120)      # 원본 해상도
    run_benchmark(resize_to=(640, 480), max_frames=120)  # 폰 카메라 전송 시 흔히 쓰는 해상도
