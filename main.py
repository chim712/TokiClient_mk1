import stt

def main():
    temp = stt.STTProcessor(aggressiveness=2, whisper_model_name="base")
    while True:
        stt.STTProcessor.record_sound_to_text(temp)


if __name__ == '__main__':
    main()