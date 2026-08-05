import asyncio

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import config
from core.detector import Detector
from core.analyzer import analyze
from core.notifier import Notifier

app = FastAPI(title="ieum AI server")

detector = Detector(
    model_path=config.MODEL_PATH,
    conf=config.CONFIDENCE,
    history_size=config.HISTORY_SIZE,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/detect")
async def detect_stream(websocket: WebSocket):
    """영상 소스를 읽어 감지 결과를 실시간으로 전송"""
    await websocket.accept()

    notifier = Notifier()
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": "영상 소스를 열 수 없습니다"})
        await websocket.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]
            detections, _ = detector.detect(frame)

            # 근접도 높은 순으로 정렬 → 가장 위험한 것부터
            analyzed = [analyze(d, frame_width, frame_height) for d in detections]
            analyzed.sort(key=lambda d: d["proximity"], reverse=True)

            for d in analyzed:
                message = notifier.build_message(d)
                if message:
                    await websocket.send_json(message)
                    break  # 한 프레임에 하나만 전송

            # 다른 작업이 돌 수 있도록 양보
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("[연결 종료] 클라이언트가 연결을 끊었습니다")
    finally:
        cap.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)