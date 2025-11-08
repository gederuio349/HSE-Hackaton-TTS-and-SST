from stt import stt_function
from tts import tts_function



text, latents = stt_function('input/rawgord.wav')
tts_function(text, latents, out_wav="output/outgord.wav")

