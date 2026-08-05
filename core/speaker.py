import io
import time
import threading

from gtts import gTTS
import pygame

import config


class Speaker:
    """감지 결과를 음성으로 안내하는 클래스"""

    def __init__(self, threshold=None, interval=None, lang=None):
        self.threshold = threshold if threshold is not None else config.PROXIMITY_THRESHOLD
        self.interval = interval if interval is not None else config.SPEAK_INTERVAL
        self.lang = lang if lang is not None else config.TTS_LANG

        # 트래킹 ID별 마지막 발화 시각
        self.last_spoken = {}
        self.is_speaking = False

        pygame.mixer.init()

    def should_speak(self, track_id, proximity):
        """근접도가 임계값 이상이고, 최근에 말한 적 없으면 True"""
        if proximity < self.threshold:
            return False
        if self.is_speaking:
            return False

        now = time.time()
        if now - self.last_spoken.get(track_id, 0) < self.interval:
            return False

        self.last_spoken[track_id] = now
        return True

    def speak(self, text):
        """텍스트를 음성으로 출력 (별도 스레드에서 재생)"""
        threading.Thread(target=self._play, args=(text,), daemon=True).start()

    def _play(self, text):
        self.is_speaking = True
        try:
            tts = gTTS(text=text, lang=self.lang)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)

            pygame.mixer.music.load(buffer)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"[TTS 오류] {e}")
        finally:
            self.is_speaking = False

    def announce(self, detection):
        """감지 결과 하나를 받아 조건에 맞으면 음성 안내"""
        if not self.should_speak(detection["track_id"], detection["proximity"]):
            return False

        text = f"{detection['direction']}에 {detection['class']}"
        self.speak(text)
        return True