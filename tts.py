from gtts import gTTS
import pygame
import time
import os
import uuid


def speak_korean(text: str) -> None:
    filename = f"temp_{uuid.uuid4().hex}.mp3"

    try:
        # TTS create
        tts = gTTS(text=text, lang='ko')
        tts.save(filename)

        # pygame init
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        # close resource
        pygame.mixer.music.stop()
        pygame.mixer.quit()

    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception as e:
                print(f"파일 삭제 중 오류: {e}")
