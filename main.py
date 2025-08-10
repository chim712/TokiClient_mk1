import stt

def main():
    temp = stt.STTProcessor(aggressiveness=2, whisper_model_name="base")
    stt.STTProcessor.transcribe_from_mic(temp)


if __name__ == '__main__':
    main()