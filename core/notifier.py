import time

import config


class Notifier:
    """어떤 감지 결과를 알릴지 판단하는 클래스 (음성 재생은 앱에서 담당)"""

    def __init__(self, threshold=None, interval=None):
        self.threshold = threshold if threshold is not None else config.PROXIMITY_THRESHOLD
        self.interval = interval if interval is not None else config.NOTIFY_INTERVAL

        # 트래킹 ID별 마지막 알림 시각
        self.last_notified = {}

    def should_notify(self, track_id, proximity):
        """근접도가 임계값 이상이고, 최근에 알린 적 없으면 True"""
        if proximity < self.threshold:
            return False

        now = time.time()
        if now - self.last_notified.get(track_id, 0) < self.interval:
            return False

        self.last_notified[track_id] = now
        return True

    def build_message(self, detection):
        """앱에 보낼 메시지 생성. 알릴 필요 없으면 None"""
        if not self.should_notify(detection["track_id"], detection["proximity"]):
            return None

        return {
            "type": "obstacle",
            "track_id": detection["track_id"],
            "class": detection["class"],
            "direction": detection["direction"],
            "proximity": detection["proximity"],
            "message": f"{detection['direction']}에 {detection['class']}",
            "timestamp": time.time(),
        }