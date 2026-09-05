# 모델 설정
MODEL_PATH = "models/yolo26n.pt"
CONFIDENCE = 0.5
HISTORY_SIZE = 10

# 입력 소스 (웹캠은 0, 영상 파일은 경로)
VIDEO_SOURCE = "data/demo.mp4"

# 알림 필터 설정
PROXIMITY_THRESHOLD = 0.6   # 이 근접도 이상만 알림
NOTIFY_INTERVAL = 3.0       # 같은 객체 재알림 최소 간격(초)

# 로컬 테스트용 (main.py, speaker.py)
SPEAK_INTERVAL = 3.0
TTS_LANG = "ko"

# 서버 설정
HOST = "0.0.0.0"
PORT = 8000

# 즉각 위험으로 볼 근접도 (이 이상이면 최우선 안내)
DANGER_PROXIMITY = 0.8