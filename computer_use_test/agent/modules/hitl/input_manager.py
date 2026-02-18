"""
HITL Input Manager - Unified input handling for Human-in-the-Loop.

Manages both voice and text input providers with automatic fallback support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from computer_use_test.agent.modules.hitl.base_input import InputMode
from computer_use_test.agent.modules.hitl.text_input import TextInputProvider
from computer_use_test.agent.modules.hitl.voice_input import STTProvider, VoiceInputProvider

if TYPE_CHECKING:
    from computer_use_test.agent.modules.hitl.chatapp_input import ChatAppInputProvider

logger = logging.getLogger(__name__)


class HITLInputManager:
    """
    Unified HITL input manager.

    Handles input from voice or text sources with automatic fallback
    from voice to text when voice input is unavailable or fails.
    """

    def __init__(
        self,
        input_mode: InputMode | str = InputMode.TEXT,
        stt_provider: STTProvider = "whisper",
        language: str = "ko",
        voice_timeout: float = 10.0,
        default_text_prompt: str = "전략을 입력하세요: ",
        chatapp_provider: ChatAppInputProvider | None = None,
    ):
        """
        Initialize the HITL input manager.

        Args:
            input_mode: Input mode ("voice", "text", "auto", or "chatapp")
            stt_provider: STT provider for voice input ("whisper", "google", "openai")
            language: Language code for STT
            voice_timeout: Maximum seconds to wait for voice input
            default_text_prompt: Default prompt for text input
            chatapp_provider: Optional ChatAppInputProvider for chat app input
        """
        # Convert string to enum if needed
        if isinstance(input_mode, str):
            self.input_mode = InputMode(input_mode)
        else:
            self.input_mode = input_mode

        # Initialize providers
        self._text_provider = TextInputProvider(default_prompt=default_text_prompt)
        self._voice_provider = VoiceInputProvider(
            stt_provider=stt_provider,
            language=language,
            timeout=voice_timeout,
        )
        self._chatapp_provider = chatapp_provider

        logger.info(f"HITLInputManager initialized: mode={self.input_mode.value}, stt={stt_provider}, voice_available={self._voice_provider.is_available()}")

    def get_input(self, prompt: str = "") -> str:
        """
        Get input based on configured mode.

        In AUTO mode, tries voice first and falls back to text on failure.

        Args:
            prompt: Optional prompt to display

        Returns:
            User input as string
        """
        if self.input_mode == InputMode.CHATAPP:
            return self._get_chatapp_input(prompt)

        elif self.input_mode == InputMode.VOICE:
            return self._get_voice_input_with_fallback(prompt)

        elif self.input_mode == InputMode.AUTO:
            return self._get_auto_input(prompt)

        else:  # TEXT
            return self._text_provider.get_input(prompt)

    def _get_voice_input_with_fallback(self, prompt: str) -> str:
        """Get voice input, falling back to text if unavailable."""
        if self._voice_provider.is_available():
            try:
                return self._voice_provider.get_input(prompt)
            except RuntimeError as e:
                logger.warning(f"Voice input failed: {e}, falling back to text")
                print(f"⚠️ 음성 입력 실패: {e}")
                print("📝 텍스트 입력으로 전환합니다.")
                return self._text_provider.get_input(prompt)
        else:
            logger.warning("Voice input unavailable, falling back to text")
            print("⚠️ 음성 입력을 사용할 수 없습니다. 텍스트로 입력해주세요.")
            return self._text_provider.get_input(prompt)

    def _get_auto_input(self, prompt: str) -> str:
        """Try voice first, fallback to text on failure."""
        if self._voice_provider.is_available():
            try:
                print("🔊 AUTO 모드: 음성 입력을 시도합니다...")
                return self._voice_provider.get_input(prompt)
            except RuntimeError as e:
                logger.warning(f"Voice input failed in AUTO mode: {e}")
                print(f"⚠️ 음성 입력 실패: {e}")
                print("📝 텍스트 입력으로 전환합니다.")
                return self._text_provider.get_input(prompt)
        else:
            # Voice not available, use text directly
            return self._text_provider.get_input(prompt)

    def _get_chatapp_input(self, prompt: str) -> str:
        """Get input from chat app, falling back to text if unavailable."""
        if self._chatapp_provider and self._chatapp_provider.is_available():
            try:
                return self._chatapp_provider.get_input(prompt)
            except RuntimeError as e:
                logger.warning(f"Chat app input failed: {e}, falling back to text")
                print(f"Chat app input failed: {e}")
                print("Falling back to text input.")
                return self._text_provider.get_input(prompt)
        else:
            logger.warning("Chat app input unavailable, falling back to text")
            print("Chat app is not connected. Falling back to text input.")
            return self._text_provider.get_input(prompt)

    def is_voice_available(self) -> bool:
        """
        Check if voice input is available.

        Returns:
            True if voice input can be used
        """
        return self._voice_provider.is_available()

    def get_current_mode(self) -> InputMode:
        """
        Get the current input mode.

        Returns:
            Current InputMode
        """
        return self.input_mode

    def set_mode(self, mode: InputMode | str) -> None:
        """
        Change the input mode.

        Args:
            mode: New input mode ("voice", "text", or "auto")
        """
        if isinstance(mode, str):
            self.input_mode = InputMode(mode)
        else:
            self.input_mode = mode
        logger.info(f"Input mode changed to: {self.input_mode.value}")

    def get_effective_mode(self) -> Literal["voice", "text", "chatapp"]:
        """
        Get the effective mode that will be used for input.

        For AUTO mode, returns the mode that would actually be used
        based on voice availability.

        Returns:
            "voice", "text", or "chatapp"
        """
        if self.input_mode == InputMode.CHATAPP:
            if self._chatapp_provider and self._chatapp_provider.is_available():
                return "chatapp"
            return "text"
        elif self.input_mode == InputMode.TEXT:
            return "text"
        elif self.input_mode == InputMode.VOICE:
            if self._voice_provider.is_available():
                return "voice"
            return "text"
        else:  # AUTO
            if self._voice_provider.is_available():
                return "voice"
            return "text"
