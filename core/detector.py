from ultralytics import YOLO
from collections import defaultdict, deque, Counter


class Detector:
    """YOLO 모델로 객체를 감지하고 트래킹 ID를 부여하는 클래스"""

    def __init__(self, model_path="models/yolo26n.pt", conf=0.5, history_size=10):
        self.model = YOLO(model_path)
        self.conf = conf
        # 트래킹 ID별 최근 N프레임의 클래스 기록 (깜빡임 방지용)
        self.class_history = defaultdict(lambda: deque(maxlen=history_size))

    def detect(self, frame):
        """
        프레임을 받아서 감지 결과 리스트와 시각화된 프레임을 반환.

        반환값:
            detections: [{track_id, class, conf, xyxy}, ...]
            annotated: 바운딩 박스가 그려진 프레임
        """
        results = self.model.track(frame, conf=self.conf, persist=True, verbose=False)
        detections = []

        for box in results[0].boxes:
            # 트래킹 ID가 아직 부여되지 않은 객체는 건너뜀
            if box.id is None:
                continue

            track_id = int(box.id)
            cls_name = self.model.names[int(box.cls)]

            self.class_history[track_id].append(cls_name)
            stable_cls = self._get_stable_class(track_id)

            detections.append({
                "track_id": track_id,
                "class": stable_cls,
                "conf": float(box.conf),
                "xyxy": box.xyxy[0].tolist(),
            })

        return detections, results[0].plot()

    def _get_stable_class(self, track_id):
        """최근 프레임 중 가장 많이 나온 클래스를 반환 (프레임마다 클래스가 흔들리는 것 방지)"""
        history = self.class_history[track_id]
        if not history:
            return None
        return Counter(history).most_common(1)[0][0]