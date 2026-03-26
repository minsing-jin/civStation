"""
Voice input provider for HITL (Human-in-the-Loop) input.

Provides microphone-based voice input with speech-to-text (STT) functionality.
Supports multiple STT backends: Whisper (local), Google, OpenAI API.
"""

import logging
import os
from typing import Literal

from civStation.agent.modules.hitl.base_input import BaseInputProvider, InputMode

logger = logging.getLogger(__name__)

STTProvider = Literal["whisper", "google", "openai"]


class VoiceInputProvider(BaseInputProvider):
    """Voice input provider using microphone and STT."""

    def __init__(
        self,
        stt_provider: STTProvider = "whisper",
        language: str = "ko",
        timeout: float = 10.0,
        phrase_time_limit: float | None = None,
    ):
        """
        Initialize the voice input provider.

        Args:
            stt_provider: Speech-to-text provider ("whisper", "google", "openai")
            language: Language code for STT (default: "ko" for Korean)
            timeout: Maximum seconds to wait for speech to start
            phrase_time_limit: Maximum seconds for a single phrase (None = no limit)
        """
        self.stt_provider = stt_provider
        self.language = language
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self._recognizer = None
        self._available = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize speech recognition library."""
        try:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
            self._available = True
            logger.info(f"VoiceInputProvider initialized with {self.stt_provider} STT")
        except ImportError:
            logger.warning("speech_recognition not installed. Install with: pip install SpeechRecognition PyAudio")
            self._available = False

    def get_input(self, prompt: str = "") -> str:
        """
        Record audio from microphone and convert to text.

        Args:
            prompt: Optional prompt to display before recording

        Returns:
            Transcribed text from speech

        Raises:
            RuntimeError: If voice input is not available or recognition fails
        """
        if not self._available:
            raise RuntimeError("Voice input not available. Install: pip install SpeechRecognition PyAudio")

        import speech_recognition as sr

        if prompt:
            print(prompt)
        print("🎤 녹음 중... (말씀하세요)")

        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Listen for speech
                audio = self._recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )

            print("🔄 음성 인식 중...")

            # Perform STT based on selected provider
            text = self._transcribe(audio)

            print(f"✅ 인식 결과: {text}")
            return text

        except sr.WaitTimeoutError as e:
            raise RuntimeError(f"음성 입력 시간 초과 ({self.timeout}초)") from e
        except sr.UnknownValueError as e:
            raise RuntimeError("음성을 인식할 수 없습니다. 다시 시도해주세요.") from e
        except sr.RequestError as e:
            raise RuntimeError(f"STT 서비스 오류: {e}") from e

    def _transcribe(self, audio) -> str:
        """
        Transcribe audio using the configured STT provider.

        Args:
            audio: AudioData from speech_recognition

        Returns:
            Transcribed text
        """
        if self.stt_provider == "whisper":
            # Use local Whisper model
            return self._recognizer.recognize_whisper(
                audio,
                language=self.language,
            )
        elif self.stt_provider == "google":
            # Use Google Web Speech API (free, no API key needed)
            return self._recognizer.recognize_google(
                audio,
                language=self.language,
            )
        elif self.stt_provider == "openai":
            # Use OpenAI Whisper API
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set for OpenAI STT")
            return self._recognizer.recognize_whisper_api(
                audio,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unknown STT provider: {self.stt_provider}")

    def is_available(self) -> bool:
        """
        Check if voice input is available.

        Returns:
            True if speech_recognition and PyAudio are installed
        """
        return self._available

    def get_mode(self) -> InputMode:
        """
        Return the input mode.

        Returns:
            InputMode.VOICE
        """
        return InputMode.VOICE
