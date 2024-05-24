import os
import sys

from PIL import Image as pil_image

from langchain_core.language_models import BaseChatModel
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

from hezar.models import Model as AsrModel

from TTS.api import TTS as _TTS

from llm_translation import translate_to_english


# IMAGE_INFORMATION_EXTRACTOR_PROMPT = \
# """
# As an image information extractor assistant, extract all the information
# from this image and return it. Then select some parts from the latter information
# that could be more relevant to the following text:

# {text}

# Your response must be in the following format:

# ### PART 1: ALL INFORMATION ###
# ...

# ### PART 2: MORE RELEVANT INFORMATION TO THE TEXT ###
# ...

# """


IMAGE_INFORMATION_EXTRACTOR_PROMPT = \
"""
As an image information extractor assistant, extract all the information
from this image and return it.
"""

class ImageInformationExtractor:

    def __init__(self, llm: BaseChatModel, device: str = 'cuda') -> None:
        self.main_llm = llm
        self.device = device

        self._processor = AutoProcessor.from_pretrained('HuggingFaceM4/idefics2-8b-AWQ')
        self.image_llm = AutoModelForVision2Seq.from_pretrained(
            'HuggingFaceM4/idefics2-8b-AWQ',
        ).to(self.device)

    def invoke(self, text: str, image_link: str, image_size: int = 224) -> str:
        image = self._resize_image(load_image(image_link), image_size)
        clean_text = IMAGE_INFORMATION_EXTRACTOR_PROMPT.format(
            text=translate_to_english(text, self.main_llm)
        )

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': clean_text},
                ]
            },
        ]

        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.image_llm.generate(**inputs, max_new_tokens=512)
        generated_texts = self._processor.batch_decode(generated_ids, skip_special_tokens=True)

        result = generated_texts[0].split('Assistant:')[1]

        return result

    def _resize_image(self, image: pil_image.Image, size: int) -> pil_image.Image:
        scale = size / min(image.size)
        if scale >= 1.:
            return image

        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        resize_image = image.resize((new_width, new_height))
        return resize_image


############################################################################
############################################################################


class PersianAutomaticSpeechRecognizer:

    def __init__(self) -> None:
        self.model = AsrModel.load('hezarai/whisper-small-fa')

    def invoke(self, audio_file_path: str) -> str:
        output = self.model.predict(audio_file_path)
        transcript = output[0]['text']
        return transcript


############################################################################
############################################################################


class TTS(_TTS):

    @property
    def is_multi_lingual(self):
        return False


class PersianTextToSpeech:

    def __init__(self) -> None:
        self.model = TTS(
            model_path="storage/models/tts/checkpoint_88000.pth",
            config_path="storage/models/tts/config.json"
        )

    def invoke(self, text: str, output_file_path: str) -> None:
        if not output_file_path.endswith('.wav'):
            output_file_path += '.wav'

        try:
            stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

            self.model.tts_to_file(text, file_path=output_file_path)
        except Exception as e:
            raise e
        finally:
            sys.stdout = stdout
