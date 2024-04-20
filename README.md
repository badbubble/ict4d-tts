# ICT4D-TTS

## usage
```python

tts = ICT4DTTS("OPENAI_API")

english_text = ("The lectures will cover theories and practice in ICT4D,"
                "  introduce case studies, highlight real-world  projects from"
                " various countries/contexts and review various tools and techniques."
                " We will discuss the iterative process of use cases and  requirements analysis."
                " Guest lectures will make you acquainted with various perspectives on ICT4D.")
# generates list of amplitude values as output
# wav = tts.english_text_to_french_speech(english_text)
# generates a wav file as output
tts.english_text_to_french_speech(english_text, is_generate_wav_file=True, file_path="sample_result.wav")

```