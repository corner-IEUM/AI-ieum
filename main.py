import cv2

import config
from core.detector import Detector
from core.analyzer import analyze
from core.speaker import Speaker


def main():
    detector = Detector(
        model_path=config.MODEL_PATH,
        conf=config.CONFIDENCE,
        history_size=config.HISTORY_SIZE,
    )
    speaker = Speaker()

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[오류] 영상을 열 수 없습니다: {config.VIDEO_SOURCE}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        detections, annotated = detector.detect(frame)

        # 근접도 높은 순으로 정렬 → 가장 위험한 것부터 안내
        analyzed = [analyze(d, frame_width, frame_height) for d in detections]
        analyzed.sort(key=lambda d: d["proximity"], reverse=True)

        for d in analyzed:
            if speaker.announce(d):
                print(f"[음성] {d['direction']}에 {d['class']} (근접도 {d['proximity']})")
                break  # 한 프레임에 하나만 안내

        cv2.imshow("ieum", annotated)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()