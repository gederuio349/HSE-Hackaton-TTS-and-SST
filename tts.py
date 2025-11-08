import torch, torchaudio, numpy as np, json
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

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


def tts_function(text, latents, out_wav="out.wav", sr=24000):
    '''
    Генерация аудиофайла моделью xtts-v2, используя эмбеддинги голоса.
    На входе текст и эмбеддинги.
    На выходе аудиофайл в формате wav
    '''
    model = load_xtts()

    gpt_cond_latent = torch.tensor(latents["gpt_cond_latent"]).to(DEVICE)
    speaker_embedding = torch.tensor(latents["speaker_embedding"]).to(DEVICE)

    try:
        out = model.inference(
            text, "ru", gpt_cond_latent, speaker_embedding,
            temperature=0.15, top_p=0.9, top_k=30
        )
    except Exception as e:
        print("Генерация голоса неуспешна по причине:", e)

    wav = torch.tensor(out["wav"]).unsqueeze(0).cpu()
    torchaudio.save(out_wav, wav, sample_rate=sr)
    print(f"Сохранено аудио: {out_wav}")
    return  out_wav
    
