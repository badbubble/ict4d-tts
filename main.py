import json
import torch
from openai import OpenAI
from TTS.api import TTS
from typing import Optional


class ICT4DTTS:
    def __init__(self, openai_api: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(self.device)

        self.base_voice_path = "voices/fr_voice.wav"
        self.openai_client = OpenAI(api_key=openai_api)

    def _translate_en_to_fr(self, text: str) -> str:
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': "system", "content": "you are a translator, translating English to French in JSON format:"
                                              " {'french': 'bonjour'}"},
                {'role': "user", "content": text},
            ],
            response_format={"type": "json_object"}
        )
        try:
            french_text = json.loads(response.choices[0].message.content)["french"]
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding error: {e}")
        except KeyError as e:
            raise KeyError(f"Key error in parsing response: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")

        return french_text

    def _generate_french_voice(self,
                               text: str,
                               is_generate_wav_file: bool = False,
                               file_path: Optional[str] = None,
                               ) -> Optional[list[float]]:

        if is_generate_wav_file:
            if file_path is None:
                raise ValueError("file_path cannot be None")
            else:
                self.tts_model.tts_to_file(text=text, speaker_wav=self.base_voice_path, language="fr",
                                           file_path=file_path)
                return
        return self.tts_model.tts(text=text, speaker_wav=self.base_voice_path, language="fr")

    def english_text_to_french_speech(self,
                                      text: str,
                                      is_generate_wav_file: bool = False,
                                      file_path: Optional[str] = None,
                                      ) -> Optional[list[float]]:

        french_text = self._translate_en_to_fr(text)
        return self._generate_french_voice(french_text, is_generate_wav_file, file_path)


if __name__ == "__main__":
    tts = ICT4DTTS("")
    english_text = ("The lectures will cover theories and practice in ICT4D,"
                    "  introduce case studies, highlight real-world  projects from"
                    " various countries/contexts and review various tools and techniques."
                    " We will discuss the iterative process of use cases and  requirements analysis."
                    " Guest lectures will make you acquainted with various perspectives on ICT4D.")
    # generates list of amplitude values as output
    # wav = tts.english_text_to_french_speech(english_text)
    # generates a wav file as output
    tts.english_text_to_french_speech(english_text, is_generate_wav_file=True, file_path="sample_result.wav")
