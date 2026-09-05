# test_demo_client.py — 기존 /ws/detect(데모, 서버가 자체적으로 영상을 읽어 push)가
# 정상적으로 프레임을 읽고 추론 결과를 보내는지 확인하는 임시 검증용 클라이언트
import asyncio
import json

import websockets

URL = "ws://localhost:8000/ws/detect"


async def main(max_messages=8, timeout_s=20):
    print(f"[연결 시도] {URL}")
    async with websockets.connect(URL) as ws:
        print("[연결됨]")
        received = 0
        try:
            async with asyncio.timeout(timeout_s):
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "error":
                        print(f"  [서버 에러] {msg['message']}")
                        return
                    print(f"  [수신 {received+1}] kind={msg.get('kind')} priority={msg.get('priority')} "
                          f"message='{msg.get('message')}'")
                    received += 1
                    if received >= max_messages:
                        break
        except TimeoutError:
            print(f"[시간초과] {timeout_s}초 동안 {received}개 수신 후 종료")
            return
    print(f"[완료] 총 {received}개 메시지 정상 수신")


if __name__ == "__main__":
    asyncio.run(main())
