import time
import wave
import os
import pyaudio
import webrtcvad
from datetime import datetime
import whisper  # Whisper STT 라이브러리 임포트
import struct  # 바이트 데이터를 숫자로 변환하기 위해 추가
import math  # RMS 계산에 필요한 수학 함수를 위해 추가


'''
파라미터 값 범위 및 특징은 각 setter 함수의 설명을 참고할 것
set_aggressiveness(self, level: int) -> None                            // VAD 민감도
set_noise_threshold_db(self, db_value: Union[int, float]) -> None       // 노이즈 기준값
set_silence_threshold_seconds(self, seconds: float) -> None             // 무음 기준 길이
set_whisper_model(self, model_name: str) -> None                        // whisper 모델 설정

record_sound_to_text(self) -> str       *Nullable                       // 요청한 함수
'''


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

    def __init__(self, aggressiveness=2, sample_rate=16000, frame_duration=30, whisper_model_name="base",
                 noise_threshold_db=-40, silence_threshold_seconds=3.0):
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
            noise_threshold_db (int): 마이크 입력의 노이즈 기준값 (dBFS).
                                      이 값보다 낮은 데시벨의 오디오는 음성이 아닌 노이즈로 간주됩니다.
                                      기본값 -40dBFS는 일반적인 환경에서 시작하기에 좋습니다.
            silence_threshold_seconds (float): 음성 감지 후 이 시간(초) 이상 무음이 지속되면 녹음을 종료합니다.
                                               기본값은 3.0초입니다.
        """
        # webrtcvad 초기화
        # aggressiveness (int):
        #   설명: webrtcvad 라이브러리에서 사용되는 음성 활동 감지(VAD)의 민감도를 설정합니다.
        #         이 값은 오디오 프레임에 음성이 포함되어 있는지 판단하는 엄격도를 나타냅니다.
        #   유형: int (정수)
        #   허용 범위: 0, 1, 2, 3
        #     - 0 (가장 덜 공격적): 가장 관대하게 음성을 감지합니다. 작은 소음도 음성으로 오인할 가능성이 높지만,
        #                           작은 목소리나 약한 발화도 놓치지 않으려는 경우에 적합합니다.
        #     - 3 (가장 공격적): 가장 엄격하게 음성을 감지합니다. 순수한 음성 신호만 음성으로 간주하며,
        #                        주변 잡음이 음성으로 오인될 가능성이 가장 낮습니다. 조용한 환경이나
        #                        정확한 음성 구간 분리가 필요할 때 적합합니다.
        #   권장 사항: 일반적으로 1 또는 2에서 시작하여 환경에 맞게 조정하는 것이 좋습니다.
        self.vad = webrtcvad.Vad(aggressiveness)
        self.aggressiveness = aggressiveness  # setter에서 이 값을 업데이트하기 위함

        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # ms
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.frame_bytes = self.frame_size * 2

        # PyAudio 초기화
        self.audio = pyaudio.PyAudio()
        self.stream = None  # 오디오 입력 스트림 객체

        # noise_threshold_db (float 또는 int):
        #   설명: 마이크로 입력되는 오디오의 데시벨(dBFS) 기준값입니다.
        #         이 값보다 낮은 음량의 소리는 webrtcvad의 판단과 관계없이
        #         음성으로 간주되지 않고 노이즈로 필터링됩니다.
        #         dBFS는 'decibels Full Scale'의 약자로, 디지털 오디오에서 0dBFS는 최대 음량을 의미하며,
        #         이 값은 항상 음수로 표현됩니다.
        #   유형: float 또는 int (실수 또는 정수)
        #   허용 범위: 일반적으로 -100 ~ 0 사이의 음수 값 (-96dBFS는 16비트 오디오의 이론적인 최소 노이즈 플로어입니다).
        #     - 값이 높아질수록 (예: -40dBFS → -30dBFS): 더 작은 소리도 음성으로 인식할 가능성이 커져 민감도가 높아집니다.
        #                                                  배경 잡음이 음성으로 포함될 수 있습니다.
        #     - 값이 낮아질수록 (예: -40dBFS → -50dBFS): 더 큰 소리만 음성으로 인식하여 잡음 필터링에 더 효과적이지만,
        #                                                  너무 작은 목소리가 무시될 수 있습니다.
        #   권장 사항: 대부분의 일반적인 환경에서는 -45 ~ -35 dBFS 범위에서 시작하여,
        #              실제 녹음 환경의 배경 잡음 수준을 고려하여 조정하는 것이 좋습니다.
        self.noise_threshold_db = noise_threshold_db

        # silence_threshold_seconds (float):
        #   설명: 음성 감지가 시작된 후, 마이크 입력이 얼마나 긴 시간(초) 동안 무음으로 지속되어야
        #         녹음을 종료할지 결정하는 기준입니다. 이 시간 동안 음성이 없으면 발화가 끝난 것으로
        #         판단하고 녹음을 마칩니다.
        #   유형: float (실수)
        #   허용 범위: 0.1 이상의 양수 값 (0에 가까울수록 짧은 무음에도 녹음 종료, 값이 클수록 긴 무음에도 녹음 지속).
        #     - 값이 작을수록 (예: 1.0초): 짧은 무음에도 빠르게 녹음을 종료합니다.
        #     - 값이 클수록 (예: 5.0초): 긴 무음 시간에도 녹음을 계속 유지하여,
        #                                사용자가 말을 하다가 잠시 멈추는 경우에도 녹음이 끊기지 않도록 합니다.
        #   권장 사항: 일반적인 대화에서는 2.0 ~ 4.0 초가 적당합니다.
        #              사용자의 발화 속도나 습관에 따라 조정합니다.
        self.silence_threshold_seconds = silence_threshold_seconds

        # whisper_model_name (str):
        #   설명: 음성을 텍스트로 변환하는 데 사용될 Whisper 모델의 크기를 지정합니다.
        #         모델 크기가 클수록 변환 정확도는 높아지지만, 모델을 로드하는 데 더 많은 시간과 메모리(RAM)가 필요하며,
        #         전사 속도도 느려질 수 있습니다.
        #   유형: str (문자열)
        #   허용 범위: "tiny", "base", "small", "medium", "large" (혹은 large-v2, large-v3 등 최신 버전)
        #   성능 고려 사항:
        #     - tiny: 가장 빠르고 가장 적은 리소스를 사용하지만, 정확도가 가장 낮습니다. 간단한 테스트나 매우 제한적인 환경에 적합합니다.
        #     - base: 속도와 정확도 사이의 좋은 균형을 제공하며, 대부분의 일반적인 사용 사례에 적합합니다. (기본값)
        #     - large: 가장 정확한 모델이지만, 가장 많은 리소스와 긴 처리 시간을 필요로 합니다. 고품질의 전사가 필요할 때 사용합니다.
        #   권장 사항: 처음 시작할 때는 "base" 모델로 테스트하고, 필요에 따라 상위 모델로 변경하여 성능을 확인하는 것이 좋습니다.
        self.whisper_model_name = whisper_model_name
        self.log(f"Whisper 모델 '{self.whisper_model_name}' 로드 중... 이 작업은 시간이 다소 소요될 수 있습니다.")
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            self.log("Whisper 모델 로드 완료.")
        except Exception as e:
            self.log(f"Whisper 모델 로드 실패: {e}. 'pip install -U openai-whisper' 및 FFmpeg 설치를 확인하세요.")
            self.whisper_model = None

            # 임시 오디오 파일을 저장할 디렉토리 설정
        self.temp_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "temp_audio")
        os.makedirs(self.temp_dir, exist_ok=True)

    def log(self, message):
        """
        콘솔에 현재 시간과 함께 메시지를 출력하는 내부 로깅 함수입니다.
        디버깅 및 사용자 피드백을 위해 사용됩니다.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def set_aggressiveness(self, level):
        """
        webrtcvad의 민감도(aggressiveness)를 설정합니다.
        이 값은 오디오 프레임에 음성이 포함되어 있는지 판단하는 엄격도를 나타냅니다.

        Args:
            level (int): VAD 민감도 수준 (0, 1, 2, 3).
                         - 0: 가장 덜 공격적 (관대)
                         - 3: 가장 공격적 (엄격)
        """
        if level in [0, 1, 2, 3]:
            self.aggressiveness = level
            self.vad = webrtcvad.Vad(self.aggressiveness)  # VAD 객체 재생성
            self.log(f"VAD 민감도(aggressiveness)가 {self.aggressiveness}로 설정되었습니다.")
        else:
            self.log(f"오류: VAD 민감도는 0, 1, 2, 3 중 하나여야 합니다. 현재 값: {level}")

    def set_noise_threshold_db(self, db_value):
        """
        마이크 입력의 노이즈 기준값(dBFS)을 설정합니다.
        이 값보다 낮은 데시벨의 오디오는 음성이 아닌 노이즈로 간주됩니다.

        Args:
            db_value (int): 새로운 노이즈 기준값 (dBFS). 일반적으로 음수.
                            - 값이 높아질수록 (예: -40dBFS → -30dBFS): 더 작은 소리도 음성으로 인식 가능.
                            - 값이 낮아질수록 (예: -40dBFS → -50dBFS): 더 큰 소리만 음성으로 인식, 잡음 필터링 강화.
        """
        if isinstance(db_value, (int, float)):
            self.noise_threshold_db = float(db_value)
            self.log(f"노이즈 기준값이 {self.noise_threshold_db} dBFS로 설정되었습니다.")
        else:
            self.log(f"오류: 노이즈 기준값은 숫자(int 또는 float)여야 합니다. 현재 값: {db_value}")

    def set_silence_threshold_seconds(self, seconds):
        """
        음성 감지 후 무음이 지속될 때 녹음을 종료할 무음 기준 길이(초)를 설정합니다.

        Args:
            seconds (float): 새로운 무음 기준 길이 (초). 0보다 커야 합니다.
                             - 값이 작을수록 (예: 1.0초): 짧은 무음에도 빠르게 녹음 종료.
                             - 값이 클수록 (예: 5.0초): 긴 무음 시간에도 녹음을 계속 유지.
        """
        if isinstance(seconds, (int, float)) and seconds > 0:
            self.silence_threshold_seconds = float(seconds)
            self.log(f"무음 기준 길이가 {self.silence_threshold_seconds} 초로 설정되었습니다.")
        else:
            self.log(f"오류: 무음 기준 길이는 0보다 큰 숫자(int 또는 float)여야 합니다. 현재 값: {seconds}")

    def set_whisper_model(self, model_name):
        """
        Whisper STT 모델을 변경하고 다시 로드합니다.
        이 함수를 호출하면 새로운 모델이 로드되므로 시간이 다소 소요될 수 있습니다.

        Args:
            model_name (str): 사용할 Whisper 모델 이름 (예: "tiny", "base", "small", "medium", "large").
        """
        allowed_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if model_name in allowed_models:
            self.whisper_model_name = model_name
            self.log(f"Whisper 모델을 '{self.whisper_model_name}'로 변경 후 로드 중... (시간 소요)")
            try:
                self.whisper_model = whisper.load_model(self.whisper_model_name)
                self.log(f"Whisper 모델 '{self.whisper_model_name}' 로드 완료.")
            except Exception as e:
                self.log(f"Whisper 모델 '{self.whisper_model_name}' 로드 실패: {e}. 이전 모델이 유지됩니다.")
                self.whisper_model = None  # 로드 실패 시 모델을 None으로 설정
        else:
            self.log(f"오류: 지원하지 않는 Whisper 모델 이름입니다. 허용된 모델: {', '.join(allowed_models)}")
            self.log(f"현재 모델: {self.whisper_model_name}")

    def _start_stream(self):
        """
        PyAudio를 사용하여 마이크로부터 오디오를 읽을 입력 스트림을 시작합니다.
        스트림이 이미 열려있지 않은 경우에만 새로운 스트림을 엽니다.
        """
        if self.stream is None:
            self.stream = self.audio.open(format=pyaudio.paInt16,
                                          channels=1,
                                          rate=self.sample_rate,
                                          input=True,
                                          frames_per_buffer=self.frame_size)
            self.log("오디오 입력 스트림 시작됨.")

    def _stop_stream(self):
        """
        현재 활성화된 오디오 입력 스트림을 중지하고 닫습니다.
        자원 누수를 방지하기 위해 스트림 사용 후 항상 호출해야 합니다.
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            self.log("오디오 입력 스트림 중지 및 닫힘.")

    def _save_frames_to_wav(self, frames):
        """
        녹음된 오디오 프레임(바이트 스트림)을 WAV 파일로 저장합니다.
        파일은 임시 디렉토리에 고유한 이름으로 저장됩니다.
        """
        filename = datetime.now().strftime("temp_recording_%Y%m%d_%H%M%S_%f.wav")
        file_path = os.path.join(self.temp_dir, filename)

        try:
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
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

    def _get_frame_rms_db(self, frame_bytes):
        """
        오디오 프레임의 RMS (Root Mean Square) 에너지를 dBFS (decibels Full Scale)로 계산합니다.
        16비트 부호 있는 PCM 오디오를 가정합니다.

        Args:
            frame_bytes (bytes): 오디오 프레임의 바이트 데이터.

        Returns:
            float: 해당 프레임의 RMS 레벨 (dBFS).
                   음량이 전혀 없는 경우 -inf를 반환합니다.
        """
        try:
            samples = struct.unpack(f'{len(frame_bytes) // 2}h', frame_bytes)
        except struct.error as e:
            self.log(f"오디오 프레임 언팩 중 오류 발생: {e}. 프레임 크기를 확인하세요.")
            return -float('inf')

        sum_squares = sum(s ** 2 for s in samples)

        if not samples:
            return -float('inf')

        mean_square = sum_squares / len(samples)
        rms = math.sqrt(mean_square)
        max_amplitude = 32767.0

        if rms == 0:
            return -96.0

        dbfs = 20 * math.log10(rms / max_amplitude)
        return dbfs

    def record_sound_to_text(self):
        """
        마이크로부터 음성을 녹음하고, 음성 활동을 감지합니다.
        음성 감지 후 self.silence_threshold_seconds 초 이상 무음이 지속되면 녹음을 종료합니다.
        녹음된 음성 파일을 Whisper를 사용하여 텍스트로 변환하고,
        임시 파일을 삭제한 후 변환된 텍스트를 반환합니다.

        반환값:
            str: 마이크로부터 인식된 텍스트.
                 음성이 전혀 감지되지 않았거나, Whisper 모델 로드 실패,
                 또는 기타 오류 발생 시 None을 반환합니다.
        """
        if self.whisper_model is None:
            self.log("Whisper 모델이 로드되지 않아 음성 인식을 수행할 수 없습니다. 초기화 메시지를 확인하세요.")
            return None

        self._start_stream()
        self.log("[STT 대기 중] 마이크에 음성을 말씀해주세요...")
        self.log(f"현재 VAD 민감도: {self.aggressiveness}")
        self.log(f"설정된 노이즈 기준값: {self.noise_threshold_db} dBFS")
        self.log(f"설정된 무음 종료 기준: {self.silence_threshold_seconds} 초")
        self.log(f"사용할 Whisper 모델: {self.whisper_model_name}")

        voiced_frames = []
        silence_start_time = None
        recording_active = False
        speech_detected_at_least_once = False

        transcribed_text = None
        temp_wav_path = None

        try:
            while True:
                frame = self.stream.read(self.frame_size, exception_on_overflow=False)

                if len(frame) < self.frame_bytes:
                    self.log("오디오 프레임 크기 부족 또는 스트림 비정상 종료 감지. 녹음 중단.")
                    break

                frame_rms_db = self._get_frame_rms_db(frame)
                vad_is_speech = self.vad.is_speech(frame, self.sample_rate)

                is_speech = vad_is_speech and (frame_rms_db > self.noise_threshold_db)

                if is_speech:
                    if not recording_active:
                        self.log("[음성 감지됨] 녹음 시작.")
                        recording_active = True
                        speech_detected_at_least_once = True

                    voiced_frames.append(frame)
                    silence_start_time = None
                else:
                    if recording_active:
                        if silence_start_time is None:
                            silence_start_time = time.time()

                        if time.time() - silence_start_time > self.silence_threshold_seconds:
                            self.log(f"[무음 감지] {self.silence_threshold_seconds}초 이상 무음 지속. 녹음 종료.")
                            break

                        voiced_frames.append(frame)

        except KeyboardInterrupt:
            self.log("\n[사용자 종료] 키보드 인터럽트 발생.")
        except Exception as e:
            self.log(f"녹음 중 예상치 못한 오류 발생: {e}")
        finally:
            self._stop_stream()

            if not speech_detected_at_least_once:
                self.log("마이크 입력에서 음성이 전혀 감지되지 않아 STT를 수행하지 않습니다.")
                return None

            if voiced_frames:
                self.log(f"총 녹음된 음성 프레임 수: {len(voiced_frames)}")
                temp_wav_path = self._save_frames_to_wav(voiced_frames)

                if temp_wav_path and self.whisper_model:
                    try:
                        self.log(f"Whisper를 사용하여 '{temp_wav_path}' 파일 전사(Transcribe) 시작...")
                        result = self.whisper_model.transcribe(temp_wav_path, fp16=False)
                        transcribed_text = result["text"]
                        self.log(f"[STT 결과] '{transcribed_text}'")
                    except Exception as e:
                        self.log(f"Whisper 전사 중 오류 발생: {e}")
                    finally:
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
    # STTProcessor 인스턴스 생성 및 기본 설정
    stt_handler = STTProcessor(
        aggressiveness=2,
        whisper_model_name="base",
        noise_threshold_db=-40,
        silence_threshold_seconds=3.0
    )

    print("\n--- 초기 설정으로 첫 번째 음성 인식 ---")
    first_text = stt_handler.record_sound_to_text()
    if first_text is not None:
        print(f"인식된 텍스트: {first_text}")
    else:
        print("첫 번째 인식 실패.")

    print("\n--- 설정 변경 후 두 번째 음성 인식 ---")
    # VAD 민감도를 가장 공격적으로 변경 (잡음에 덜 민감)
    stt_handler.set_aggressiveness(3)
    # Whisper 모델을 "tiny"로 변경 (빠른 테스트용, 정확도 낮음)
    stt_handler.set_whisper_model("tiny")
    # 노이즈 기준값을 -30dBFS로 변경 (더 작은 소리도 음성으로 인식)
    stt_handler.set_noise_threshold_db(-30)
    # 무음 종료 기준을 5초로 변경 (말을 오래 멈춰도 녹음 유지)
    stt_handler.set_silence_threshold_seconds(5.0)

    second_text = stt_handler.record_sound_to_text()
    if second_text is not None:
        print(f"인식된 텍스트: {second_text}")
    else:
        print("두 번째 인식 실패.")

    # PyAudio 리소스 정리
    stt_handler.audio.terminate()
