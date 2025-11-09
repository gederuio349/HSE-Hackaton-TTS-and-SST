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


def convert_json_to_tensors(emb_list, cond_list):
    """
    Конвертирует списки из JSON обратно в torch тензоры с правильной формой.
    """
    emb_array = np.array(emb_list, dtype=np.float32)
    cond_array = np.array(cond_list, dtype=np.float32)
    
    speaker_embedding = torch.from_numpy(emb_array).to(DEVICE)
    gpt_cond_latent = torch.from_numpy(cond_array).to(DEVICE)
    
    if len(speaker_embedding.shape) == 1:
        # [features] -> [1, features, 1]
        speaker_embedding = speaker_embedding.unsqueeze(0).unsqueeze(-1)
    elif len(speaker_embedding.shape) == 2:
        if speaker_embedding.shape[0] < speaker_embedding.shape[1]:
            # [1, features] -> [1, features, 1]
            speaker_embedding = speaker_embedding.unsqueeze(-1)
        else:
            # [features, 1] -> [1, features, 1]
            speaker_embedding = speaker_embedding.unsqueeze(0)
    elif len(speaker_embedding.shape) == 3:
        if speaker_embedding.shape[0] != 1:
            speaker_embedding = speaker_embedding[0:1]

    if len(gpt_cond_latent.shape) == 2:
        # [channels, time] -> [1, channels, time]
        gpt_cond_latent = gpt_cond_latent.unsqueeze(0)
    elif len(gpt_cond_latent.shape) == 3:
        if gpt_cond_latent.shape[0] != 1:
            gpt_cond_latent = gpt_cond_latent[0:1]
            
    return speaker_embedding, gpt_cond_latent


def tts_function(text, emb, cond, out_wav="out.wav", sr=24000):
    '''
    Генерация аудиофайла моделью xtts-v2, используя эмбеддинги голоса.
    На входе текст и эмбеддинги.
    На выходе аудиофайл в формате wav
    '''
    model = load_xtts()

    # Преобразуем векторы из JSON формата в тензоры
    speaker_embedding, gpt_cond_latent = convert_json_to_tensors(emb, cond)

    try:
        out = model.inference(
            text, "ru", gpt_cond_latent, speaker_embedding,
            temperature=0.15, top_p=0.9, top_k=30
        )
        
        wav = torch.tensor(out["wav"]).unsqueeze(0).cpu()
        torchaudio.save(out_wav, wav, sample_rate=sr)
        print(f"Сохранено аудио: {out_wav}")
        return out_wav
    except Exception as e:
        print("Генерация голоса неуспешна по причине:", e)
        raise  
    
