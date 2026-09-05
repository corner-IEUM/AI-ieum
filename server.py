# server.py — ieum AI 서버 (FastAPI). 두 개의 WebSocket 엔드포인트를 제공한다.
#
# /ws/detect(데모 모드): "사전 계산된 detections.json을 재생하는 방식"이 아니다 — 그런 파일 자체가
# 이 프로젝트에 없다. 실제로는 연결이 들어올 때마다 cv2.VideoCapture(config.VIDEO_SOURCE)로
# data/demo.mp4를 처음부터 다시 열어, 프레임을 읽는 즉시 그 자리에서 YOLO 추론을 돌려 결과를
# 보낸다(라이브 추론) — 프레임을 미리 계산해두는 캐시가 전혀 없다. 연결마다 독립된 Detector/
# VideoCapture 인스턴스를 새로 만드므로(아래 detect_stream 참고), 데모 세션끼리 상태가 섞이지 않는다.
#
# ※ 성능 트레이드오프: benchmark_detector.py로 실측한 결과 이 YOLO 모델의 순수 추론 속도는
# 프레임당 평균 약 71ms(≈14fps) — data/demo.mp4 자체는 30fps/약 282초 길이라, 감지 파이프라인이
# 영상 재생 속도보다 약 2.1배 느리다. 프론트(DemoVideoFeed.tsx)가 재생하는 화면(/media/demo.mp4,
# 자체 속도로 재생)과 이 엔드포인트가 만들어내는 감지 결과(음성 안내)는 완전히 별개의 소비자라
# 서로 동기화되지 않으므로, 데모가 길어질수록 "화면에 보이는 장면"과 "음성으로 안내되는 감지
# 결과" 사이의 시간차가 누적된다(초반 수십 초는 체감하기 어렵지만, 감지 쪽이 원본 영상 재생 시간
# 대비 약 2.1배 걸리므로 세션이 길어질수록 어긋남이 커짐).
#
# /ws/detect/live(실시간 모드): 클라이언트(폰 카메라)가 보내주는 프레임을 그 자리에서 추론만 한다.
import asyncio
import contextlib

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import config
from core.detector import Detector
from core.analyzer import analyze
from core.notifier import Notifier
from core.live_detector import LiveDetector, decode_base64_jpeg

app = FastAPI(title="ieum AI server")

# 프론트엔드 데모 모드 화면에서 demo.mp4를 실제로 재생해서 보여주기 위해 정적 파일로도 노출한다.
# (감지 자체는 여전히 cv2.VideoCapture로 서버가 직접 읽어서 처리 — 이건 순전히 화면 표시용)
app.mount("/media", StaticFiles(directory="data"), name="media")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/detect")
async def detect_stream(websocket: WebSocket):
    """영상 소스를 읽어 감지 결과를 실시간으로 전송"""
    await websocket.accept()

    # 연결마다 독립된 Detector를 새로 만든다 — 예전엔 모듈 전역 detector 하나를 모든 연결이
    # 공유했는데, 백엔드(AiWebSocketClient)가 안내 기능용으로 항상 /ws/detect에 붙어있는 상태에서
    # 프론트엔드 데모 모드가 동시에 접속하면, persist=True 트래커가 서로 다른 두 영상 재생
    # 위치의 프레임을 번갈아 받아 상태가 뒤섞여 감지가 멈추는 문제를 직접 재현해서 확인했다.
    # /ws/detect/live가 연결마다 LiveDetector를 새로 만드는 것과 같은 이유·같은 해결책.
    detector = Detector(
        model_path=config.MODEL_PATH,
        conf=config.CONFIDENCE,
        history_size=config.HISTORY_SIZE,
    )
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

            # 추론(계산)과 전송(send)을 반드시 분리한다 — send_json은 try 블록 밖에서만 호출해야
            # 클라이언트가 이미 끊긴 상태에서의 전송 실패(WebSocketDisconnect)가 아래 except Exception에
            # 잡혀 "이미 닫힌 소켓에 에러 메시지를 또 보내려다 RuntimeError로 서버 전체가 죽는" 사고를
            # 막을 수 있다 (실제로 이 버그로 AI 서버가 죽는 것을 재현해서 확인함 — send를 try 안에 두면
            # WebSocketDisconnect가 Exception에 걸려 마스킹되고, 그 상태에서 또 send_json을 호출하다
            # "Cannot call send once a close message has been sent" RuntimeError가 나서 아래
            # except WebSocketDisconnect로도 못 잡히고 서버 프로세스가 죽는다).
            message = None
            try:
                frame_height, frame_width = frame.shape[:2]
                detections, _ = detector.detect(frame)

                # 근접도 높은 순으로 정렬 → 가장 위험한 것부터
                analyzed = [analyze(d, frame_width, frame_height) for d in detections]
                analyzed.sort(key=lambda d: d["proximity"], reverse=True)

                for d in analyzed:
                    candidate = notifier.build_message(d)
                    if candidate:
                        message = candidate
                        break  # 한 프레임에 하나만 전송
            except Exception as e:
                # 추론 도중 예외가 나도 소켓은 유지하고 다음 프레임을 계속 받는다 (/ws/detect/live와 동일)
                message = {"type": "error", "message": f"추론 실패: {e}"}

            if message:
                await websocket.send_json(message)  # 여기서 끊기면 WebSocketDisconnect가 아래로 그대로 전파됨

            # 다른 작업이 돌 수 있도록 양보
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("[연결 종료] 클라이언트가 연결을 끊었습니다")
    finally:
        cap.release()


@app.websocket("/ws/detect/live")
async def detect_live(websocket: WebSocket):
    """폰 카메라 등 클라이언트가 보내는 프레임(base64 JPEG)을 받아 즉시 YOLO 추론 후
    같은 소켓으로 결과를 돌려준다. 기존 /ws/detect(로컬 영상 재생)와는 별개로 동작하며
    응답 JSON 포맷은 동일하게 notifier.build_message()를 그대로 재사용한다."""
    await websocket.accept()

    # 연결마다 독립된 트래커 상태를 갖도록 세션별로 새로 생성
    # (persist=True인 YOLO 트래커는 프레임이 이어져야 track_id가 유지되므로,
    #  같은 연결의 프레임들은 반드시 같은 LiveDetector 인스턴스를 통과해야 함)
    live_detector = LiveDetector(
        model_path=config.MODEL_PATH,
        conf=config.CONFIDENCE,
        history_size=config.HISTORY_SIZE,
    )
    notifier = Notifier()

    # 수신 루프와 처리 루프를 분리: 처리가 프레임 도착 속도를 못 따라가도
    # latest_frame이 항상 '가장 최근에 도착한 프레임'으로 덮어써지기 때문에
    # 오래된 프레임이 큐에 쌓여 지연이 누적되는 일이 없다 (최신 프레임 우선 처리)
    latest_frame: dict = {"data": None}
    disconnected = asyncio.Event()

    async def receive_loop():
        try:
            while True:
                data = await websocket.receive_text()
                latest_frame["data"] = data
        except WebSocketDisconnect:
            disconnected.set()

    receiver_task = asyncio.create_task(receive_loop())

    try:
        while not disconnected.is_set():
            data = latest_frame["data"]
            if data is None:
                await asyncio.sleep(0.005)
                continue
            latest_frame["data"] = None  # 소비 표시 — 같은 프레임을 중복 처리하지 않음

            frame = decode_base64_jpeg(data)
            if frame is None:
                # 잘못된 프레임이 와도 연결은 끊지 않고 에러만 알린 뒤 계속 수신 대기
                await websocket.send_json({"type": "error", "message": "프레임 디코딩 실패"})
                continue

            # 추론과 전송을 분리하는 이유는 /ws/detect와 동일 — send_json이 try 안에 있으면
            # 클라이언트 연결 종료로 인한 전송 실패가 except Exception에 잡혀 이미 닫힌 소켓에
            # 또 send_json을 시도하다 RuntimeError로 서버가 죽을 수 있다(직접 재현해서 확인).
            message = None
            try:
                frame_height, frame_width = frame.shape[:2]
                detections, _ = live_detector.detect(frame)

                analyzed = [analyze(d, frame_width, frame_height) for d in detections]
                analyzed.sort(key=lambda d: d["proximity"], reverse=True)

                for d in analyzed:
                    candidate = notifier.build_message(d)
                    if candidate:
                        message = candidate
                        break  # 기존 /ws/detect와 동일하게 프레임당 하나만 전송
            except Exception as e:
                # 추론 도중 예외가 나도 소켓은 유지하고 다음 프레임을 계속 받는다
                message = {"type": "error", "message": f"추론 실패: {e}"}

            if message:
                await websocket.send_json(message)  # 여기서 끊기면 WebSocketDisconnect가 바깥으로 그대로 전파됨
    except WebSocketDisconnect:
        pass
    finally:
        receiver_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver_task


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)