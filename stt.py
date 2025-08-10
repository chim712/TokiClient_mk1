import time
import wave
import os
import pyaudio
import webrtcvad
from datetime import datetime
import whisper  # Whisper STT 라이브러리 임포트


class STTProcessor:
    """
    마이크로부터 음성을 녹음하고, 음성 활동 감지(VAD)를 통해
    음성을 텍스트로 변환하는 기능을 제공하는 모듈입니다.

    요구사항:
    1. webrtcvad: 음성 활동 감지 (pip install webrtcvad)
    2. PyAudio: 마이크 입력 처리 (pip install pyaudio)
    3. Whisper: 음성-텍스트 변환 (pip install openai-whisper)
    4. FFmpeg: 시스템 PATH에 설치되어 있어야 함 (Whisper가 내부적으로 사용).
    """

    def __init__(self, aggressiveness=2, sample_rate=16000, frame_duration=30, whisper_model_name="base"):
        """
        STTProcessor를 초기화합니다.

        Args:
            aggressiveness (int): webrtcvad의 민감도 (0: 가장 덜 공격적, 3: 가장 공격적).
                                  높을수록 음성이 아닌 소리를 음성으로 잘못 분류할 가능성이 줄어듭니다.
            sample_rate (int): 오디오 샘플링 속도 (Hz). Whisper는 16000Hz를 권장합니다.
            frame_duration (int): VAD가 오디오를 분석하는 프레임의 길이 (ms).
                                  webrtcvad는 10, 20, 30ms 프레임만 지원합니다.
            whisper_model_name (str): 사용할 Whisper 모델 이름 (예: "tiny", "base", "small", "medium", "large").
                                      모델 크기가 클수록 정확도는 높지만 로드 시간과 메모리 사용량이 증가합니다.
        """
        # webrtcvad 초기화
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # ms
        # 프레임당 샘플 수 계산: (샘플링 속도 * 프레임 길이) / 1000ms
        self.frame_size = int(sample_rate * frame_duration / 1000)
        # 16비트 오디오는 샘플당 2바이트이므로 프레임당 바이트 수 계산
        self.frame_bytes = self.frame_size * 2

        # PyAudio 초기화
        # 오디오 스트림을 관리하기 위한 PyAudio 인스턴스입니다.
        self.audio = pyaudio.PyAudio()
        self.stream = None  # 오디오 입력 스트림 객체

        # Whisper 모델 로드 (초기화 시 한 번만 로드하여 효율성 증대)
        self.log(f"Whisper 모델 '{whisper_model_name}' 로드 중... 이 작업은 시간이 다소 소요될 수 있습니다.")
        try:
            # Whisper 모델 로드. CPU 사용 시 fp16=False를 명시하여 정확도를 높일 수 있습니다.
            self.whisper_model = whisper.load_model(whisper_model_name)
            self.log("Whisper 모델 로드 완료.")
        except Exception as e:
            self.log(f"Whisper 모델 로드 실패: {e}. 'pip install -U openai-whisper' 및 FFmpeg 설치를 확인하세요.")
            self.whisper_model = None  # 모델 로드 실패 시 None으로 설정하여 에러 처리

        # 임시 오디오 파일을 저장할 디렉토리 설정
        # 현재 스크립트 파일이 실행되는 디렉토리 내에 'temp_audio' 폴더를 생성합니다.
        self.temp_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "temp_audio")
        os.makedirs(self.temp_dir, exist_ok=True)  # 폴더가 없으면 생성, 이미 있으면 무시

    def log(self, message):
        """
        콘솔에 현재 시간과 함께 메시지를 출력하는 내부 로깅 함수입니다.
        디버깅 및 사용자 피드백을 위해 사용됩니다.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _start_stream(self):
        """
        PyAudio를 사용하여 마이크로부터 오디오를 읽을 입력 스트림을 시작합니다.
        스트림이 이미 열려있지 않은 경우에만 새로운 스트림을 엽니다.
        """
        if self.stream is None:
            self.stream = self.audio.open(format=pyaudio.paInt16,  # 16비트 정수 형식의 오디오
                                          channels=1,  # 모노 채널
                                          rate=self.sample_rate,  # 설정된 샘플링 속도
                                          input=True,  # 입력(마이크) 스트림
                                          frames_per_buffer=self.frame_size)  # 버퍼당 읽을 프레임 수
            self.log("오디오 입력 스트림 시작됨.")

    def _stop_stream(self):
        """
        현재 활성화된 오디오 입력 스트림을 중지하고 닫습니다.
        자원 누수를 방지하기 위해 스트림 사용 후 항상 호출해야 합니다.
        """
        if self.stream:
            self.stream.stop_stream()  # 스트림 중지
            self.stream.close()  # 스트림 닫기
            self.stream = None  # 스트림 객체 초기화
            self.log("오디오 입력 스트림 중지 및 닫힘.")

    def _save_frames_to_wav(self, frames):
        """
        녹음된 오디오 프레임(바이트 스트림)을 WAV 파일로 저장합니다.
        파일은 임시 디렉토리에 고유한 이름으로 저장됩니다.
        """
        # 현재 타임스탬프를 포함하여 고유한 파일 이름 생성 (밀리초 포함)
        filename = datetime.now().strftime("temp_recording_%Y%m%d_%H%M%S_%f.wav")
        file_path = os.path.join(self.temp_dir, filename)  # 임시 파일의 전체 경로

        try:
            # wave 모듈을 사용하여 WAV 파일 생성 및 데이터 쓰기
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)  # 채널 수: 모노
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))  # 샘플 폭: 16비트
                wf.setframerate(self.sample_rate)  # 프레임 속도 (샘플링 속도)
                wf.writeframes(b''.join(frames))  # 모든 프레임을 연결하여 파일에 쓰기
            self.log(f"임시 WAV 파일 저장 완료: {file_path}")
            return file_path
        except Exception as e:
            self.log(f"임시 WAV 파일 저장 중 오류 발생: {e}")
            return None

    def _delete_file(self, file_path):
        """
        지정된 경로의 파일을 안전하게 삭제합니다.
        파일이 존재하지 않거나 삭제 중 오류가 발생하면 로그를 남깁니다.
        """
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                self.log(f"임시 파일 삭제 완료: {file_path}")
            except Exception as e:
                self.log(f"임시 파일 삭제 중 오류 발생: {e}")

    def transcribe_from_mic(self):
        """
        마이크로부터 음성을 녹음하고, 음성 활동을 감지합니다.
        음성 감지 후 3초 이상 무음이 지속되면 녹음을 종료합니다.
        녹음된 음성 파일을 Whisper를 사용하여 텍스트로 변환하고,
        임시 파일을 삭제한 후 변환된 텍스트를 반환합니다.

        반환값:
            str: 마이크로부터 인식된 텍스트.
                 음성이 전혀 감지되지 않았거나, Whisper 모델 로드 실패,
                 또는 기타 오류 발생 시 None을 반환합니다.
        """
        # Whisper 모델이 성공적으로 로드되었는지 확인
        if self.whisper_model is None:
            self.log("Whisper 모델이 로드되지 않아 음성 인식을 수행할 수 없습니다. 초기화 메시지를 확인하세요.")
            return None

        # 오디오 스트림 시작
        self._start_stream()
        self.log("[STT 대기 중] 마이크에 음성을 말씀해주세요...")

        voiced_frames = []  # 음성 또는 음성 후 무음 프레임을 저장할 리스트
        silence_start_time = None  # 무음이 시작된 시간 (타이머)
        recording_active = False  # 음성이 감지되어 녹음이 활성화된 상태인지 여부
        speech_detected_at_least_once = False  # 이번 호출에서 최소 한 번이라도 음성이 감지되었는지 여부

        transcribed_text = None  # 최종 변환될 텍스트
        temp_wav_path = None  # 임시 WAV 파일 경로

        try:
            while True:
                # 오디오 스트림에서 한 프레임(self.frame_size 만큼) 읽기
                # exception_on_overflow=False: 버퍼 오버플로우 시 예외를 발생시키지 않고 계속 진행
                frame = self.stream.read(self.frame_size, exception_on_overflow=False)

                # 읽은 프레임의 길이가 예상보다 짧으면 (예: 마이크 연결 끊김) 루프 종료
                if len(frame) < self.frame_bytes:
                    self.log("오디오 프레임 크기 부족 또는 스트림 비정상 종료 감지. 녹음 중단.")
                    break

                # webrtcvad를 사용하여 현재 프레임에 음성이 있는지 확인
                is_speech = self.vad.is_speech(frame, self.sample_rate)

                if is_speech:
                    # 음성이 감지됨
                    if not recording_active:
                        # 아직 녹음이 활성화되지 않았다면, 지금부터 녹음 시작으로 간주
                        self.log("[음성 감지됨] 녹음 시작.")
                        recording_active = True
                        speech_detected_at_least_once = True  # 최소 한 번은 음성 감지됨 표시

                    voiced_frames.append(frame)  # 현재 프레임을 녹음 데이터에 추가
                    silence_start_time = None  # 음성이 감지되었으니 무음 타이머 초기화
                else:
                    # 무음이 감지됨
                    if recording_active:
                        # 녹음이 이미 활성화된 상태에서 무음이 감지된 경우 (말하다가 멈춘 경우)
                        if silence_start_time is None:
                            # 무음 타이머가 시작되지 않았다면 현재 시간을 무음 시작 시간으로 설정
                            silence_start_time = time.time()

                        # 무음이 3초 이상 지속되었는지 확인
                        if time.time() - silence_start_time > 2.0:
                            self.log("[무음 감지] ?초 이상 무음 지속. 녹음 종료.")
                            break  # 3초 무음 조건 충족, 녹음 루프 종료

                        # 무음 프레임도 녹음 데이터에 포함 (음성 끝부분이 잘리지 않도록)
                        voiced_frames.append(frame)
                    # else: recording_active가 False인 상태에서는 (음성이 전혀 없었던 초기 상태)
                    #     무음이 계속되더라도 녹음을 시작하지 않고 계속 음성 대기.
                    #     이 부분이 "말을 하다가 무음이 되는 것에 한정하도록 하자" 요구사항을 만족시킵니다.

        except KeyboardInterrupt:
            # 사용자가 Ctrl+C를 눌러 프로그램을 강제 종료한 경우
            self.log("\n[사용자 종료] 키보드 인터럽트 발생.")
        except Exception as e:
            # 녹음 또는 VAD 처리 중 예상치 못한 오류 발생 시
            self.log(f"녹음 중 예상치 못한 오류 발생: {e}")
        finally:
            # 어떤 상황에서도 오디오 스트림을 안전하게 중지하고 닫기
            self._stop_stream()

            # 최소 한 번이라도 음성이 감지되지 않았다면 STT를 수행하지 않고 None 반환
            if not speech_detected_at_least_once:
                self.log("마이크 입력에서 음성이 전혀 감지되지 않아 STT를 수행하지 않습니다.")
                return None

            # 녹음된 음성 프레임이 있는 경우에만 처리
            if voiced_frames:
                self.log(f"총 녹음된 음성 프레임 수: {len(voiced_frames)}")
                # 녹음된 프레임을 WAV 파일로 저장
                temp_wav_path = self._save_frames_to_wav(voiced_frames)

                if temp_wav_path and self.whisper_model:
                    try:
                        self.log(f"Whisper를 사용하여 '{temp_wav_path}' 파일 전사(Transcribe) 시작...")
                        # Whisper 모델의 transcribe 함수는 오디오 파일 경로를 직접 처리합니다.
                        # fp16=False는 CPU 사용 시 메모리 사용량과 성능의 균형을 맞추는 데 도움이 됩니다.
                        result = self.whisper_model.transcribe(temp_wav_path, fp16=False, language="ko")
                        transcribed_text = result["text"]  # 결과에서 텍스트 추출
                        self.log(f"[STT 결과] '{transcribed_text}'")
                    except Exception as e:
                        self.log(f"Whisper 전사 중 오류 발생: {e}")
                    finally:
                        # 전사 완료 후 임시 파일 삭제
                        self._delete_file(temp_wav_path)
                else:
                    self.log("WAV 파일 저장 또는 Whisper 모델 로드에 문제가 있어 전사를 수행할 수 없습니다.")
            else:
                self.log("유효한 음성 프레임이 녹음되지 않았습니다.")

        return transcribed_text


# ==============================================================================
# 모듈 사용 예시 (단일 파일 실행을 위한 메인 진입점)
# ==============================================================================
if __name__ == "__main__":
    # STTProcessor 인스턴스 생성
    # aggressiveness는 VAD의 민감도를 조절합니다 (0: 낮음 ~ 3: 높음).
    # "base" 모델은 대부분의 경우 좋은 정확도와 속도 균형을 제공합니다.
    stt_handler = STTProcessor(aggressiveness=2, whisper_model_name="base")

    # 진입 함수 호출
    final_text = stt_handler.transcribe_from_mic()

    if final_text is not None:
        print("\n--- 최종 마이크 음성 인식 결과 ---")
        print(final_text)
        print("------------------------------")
    else:
        print("\n음성 인식을 완료하지 못했거나 유효한 음성이 감지되지 않았습니다.")

    # PyAudio 리소스 정리
    # STTProcessor 인스턴스가 더 이상 필요 없으면 audio.terminate()를 호출하여
    # PyAudio 시스템을 종료하고 모든 리소스를 해제합니다.
    # 만약 stt_handler.transcribe_from_mic()를 여러 번 호출할 계획이라면,
    # 이 terminate() 호출은 애플리케이션의 최종 종료 시점에 한 번만 실행되도록 조정해야 합니다.
    stt_handler.audio.terminate()
