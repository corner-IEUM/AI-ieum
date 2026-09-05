# test_live_client.py — /ws/detect/live 테스트용 클라이언트
# 웹캠이 없는 환경에서도 테스트할 수 있도록 기본은 data/30s.mp4 프레임을 사용하고,
# --webcam 옵션을 주면 실제 웹캠(장치 0)에서 프레임을 읽는다.
#
# 사용법 (ai/ 루트 디렉토리에서 실행):
#   python tests/test_live_client.py            # 영상 파일로 테스트
#   python tests/test_live_client.py --webcam   # 실제 웹캠으로 테스트
import argparse
import asyncio
import base64
import json
import time

import cv2
import websockets

SERVER_URL = "ws://localhost:8000/ws/detect/live"
VIDEO_PATH = "data/30s.mp4"


def encode_frame(frame, resize_to=(640, 480), quality=80) -> str:
    """OpenCV 프레임을 base64 JPEG 문자열로 인코딩 (폰 카메라 전송을 흉내)"""
    if resize_to:
        frame = cv2.resize(frame, resize_to)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def receive_loop(ws, sent_times: dict):
    """서버 응답을 계속 받아서, 알림이 온 경우 지연시간과 함께 출력"""
    async for raw in ws:
        recv_at = time.perf_counter()
        msg = json.loads(raw)
        if msg.get("type") == "error":
            print(f"  [서버 에러] {msg['message']}")
            continue
        # 알림 메시지 — 가장 최근에 보낸 프레임 기준으로 대략적 지연시간 표시
        latency_ms = (recv_at - sent_times.get("last", recv_at)) * 1000
        print(f"  [수신] kind={msg.get('kind')} priority={msg.get('priority')} "
              f"message='{msg.get('message')}' (약 {latency_ms:.0f}ms 전 프레임 기준)")


async def send_loop(ws, source, max_frames, target_interval, sent_times: dict):
    sent = 0
    while sent < max_frames:
        ret, frame = source.read()
        if not ret:
            break
        try:
            b64 = encode_frame(frame)
            await ws.send(b64)
            sent_times["last"] = time.perf_counter()
        except Exception as e:
            print(f"  [전송 오류] {e}")
        sent += 1
        await asyncio.sleep(target_interval)
    print(f"[완료] 프레임 {sent}개 전송")


async def send_bad_frame(ws):
    """예외처리(요구사항 7) 확인용 — 디코딩 불가능한 값을 한 번 보내본다"""
    await ws.send("이건-유효한-base64-JPEG가-아님")


async def main(use_webcam: bool, max_frames: int, fps: float):
    source = cv2.VideoCapture(0 if use_webcam else VIDEO_PATH)
    if not source.isOpened():
        print("[오류] 프레임 소스를 열 수 없습니다")
        return

    async with websockets.connect(SERVER_URL, max_size=10 * 1024 * 1024) as ws:
        print(f"[연결됨] {SERVER_URL}")
        sent_times: dict = {}

        await send_bad_frame(ws)  # 잘못된 프레임 먼저 하나 보내서 에러 응답 확인

        recv_task = asyncio.create_task(receive_loop(ws, sent_times))
        t0 = time.perf_counter()
        await send_loop(ws, source, max_frames, 1 / fps, sent_times)
        elapsed = time.perf_counter() - t0
        print(f"[요약] {max_frames}프레임 전송에 {elapsed:.2f}초 소요 (목표 {fps} FPS)")

        await asyncio.sleep(1.0)  # 마지막 프레임의 응답을 기다려줌
        recv_task.cancel()

    source.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true", help="영상 파일 대신 실제 웹캠(장치 0) 사용")
    parser.add_argument("--frames", type=int, default=60, help="전송할 프레임 수")
    parser.add_argument("--fps", type=float, default=10.0, help="전송 목표 FPS")
    args = parser.parse_args()

    asyncio.run(main(args.webcam, args.frames, args.fps))
