from typing import Dict

from agent import Agent
from helpers_multimodal import (
    ImageInformationExtractor,
    PersianAutomaticSpeechRecognizer,
    PersianTextToSpeech,
)


class MultimodalAgent(Agent):

    def __init__(self) -> None:
        super().__init__()

        self.iie = ImageInformationExtractor(self.llm)
        self.asr = PersianAutomaticSpeechRecognizer()
        self.tts = PersianTextToSpeech()

    def run(
        self,
        config: Dict,
        question_text: str = "",
        question_audio_path: str = "",
        question_image_path: str = "",
        answer_audio_path: str = "assets/answer.wav",
        reset_db: bool = True,
        clear_message_history: bool = True,
    ) -> None:
        assert question_text or question_audio_path, "text and audio cannot be both empty"
        assert not (question_text and question_audio_path), "text and audio cannot be both filled"

        if question_audio_path:
            question_text = self.asr.invoke(question_audio_path)

        if question_image_path:
            image_info = self.iie.invoke(question_text, question_image_path)
            question_text += "\n\n[Here is the attached image's information as text:]\n" + image_info

        super().run(question_text, config, reset_db, clear_message_history)

        answer = self._graph.get_state(config).values['messages'][-1].content
        self.tts.invoke(answer, answer_audio_path)
