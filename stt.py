import torch, torchaudio, numpy as np, json
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import whisper


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR='xtts'
CONFIG_PATH = f"{CKPT_DIR}/config.json"


def load_xtts():
    '''
    Загрузка модели xtts-v2
    '''
    cfg = XttsConfig()
    cfg.load_json(CONFIG_PATH)
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=CKPT_DIR, use_deepspeed=False)
    model.to(DEVICE)
    model.eval()
    return model


def precompute_xtts_conditioning(ref_wav_path):
    '''
    Функция для извлечения эмбеддинга голоса(speaker_embedding) + эмбеддинга манеры произношения (gpt_cond_latent) 
    '''
    model = load_xtts()

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[str(ref_wav_path)]
    )

    latents = {
        "gpt_cond_latent": gpt_cond_latent.to(DEVICE, dtype=torch.float32),
        "speaker_embedding": speaker_embedding.to(DEVICE, dtype=torch.float32),
    }
    return latents


def load_whisper_model(model_name):
    """
    Возвращает Whisper-модель, загружая при первом обращении.
    """
    whisper_model = whisper.load_model(model_name)
    return whisper_model


def transcribe_file(file_path: str, language: str = 'ru', model_name: str = "large-v3"):
    """
    Транскрибирует аудио и возвращает текст с помощью whisper
    """
    model = load_whisper_model(model_name)

    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Ресемплируем до 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Преобразуем в numpy массив (моно канал)
    audio = waveform.squeeze().numpy()

    options = {}
    if language:
        options["language"] = language

    result = model.transcribe(audio, **options)
    text = result.get("text")
    return text


def stt_function(path_to_wav_file):
    '''
    На входе wav файл.
    Ны выходе текст и cловарь с эмбеддингами.
    '''
    # Получаем эмбеддинги голоса
    latents = precompute_xtts_conditioning(path_to_wav_file)

    # Получаем транскрибированный текст
    text = transcribe_file(path_to_wav_file)
    
    return text, latents

    

if __name__ == "__main__":
    stt_function('input/raw.wav')