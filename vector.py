import torch
import torch.nn.functional as F
import json


def cosine_distance(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """
    Вычисляет косинусное расстояние между двумя эмбеддингами.
    
    Args:
        embedding1: Первый эмбеддинг (torch.Tensor)
        embedding2: Второй эмбеддинг (torch.Tensor)
    
    Returns:
        float: Косинусное расстояние между эмбеддингами
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(f"Эмбеддинги должны иметь одинаковую форму. "
                        f"Получено: {embedding1.shape} и {embedding2.shape}")
    
    # Преобразуем в одномерные векторы
    emb1_flat = embedding1.flatten()
    emb2_flat = embedding2.flatten()

    cosine_sim = F.cosine_similarity(emb1_flat.unsqueeze(0), emb2_flat.unsqueeze(0), dim=1)
    cosine_dist = 1 - cosine_sim.item()

    return cosine_dist



def compare_speakers(speaker_embedding1: torch.Tensor, speaker_embedding2: torch.Tensor, 
                     threshold: float = 0.3) -> dict:
    """
    Сравнивает два speaker_embedding и определяет, принадлежат ли они одному говорящему.
    
    Args:
        speaker_embedding1: Первый speaker_embedding
        speaker_embedding2: Второй speaker_embedding
        threshold: Пороговое значение косинусного расстояния для определения одного говорящего.
                  По умолчанию 0.3 (эмбеддинги с расстоянием < 0.3 считаются одним говорящим)
    
    Returns:
        dict: Словарь с результатами сравнения:
            - 'distance': косинусное расстояние
            - 'is_same_speaker': True если расстояние < threshold
            - 'similarity': косинусное сходство (1 - distance)
    """
    distance = cosine_distance(speaker_embedding1, speaker_embedding2)
    similarity = 1 - distance
    
    result = {
        'distance': distance,
        'similarity': similarity,
        'is_same_speaker': distance < threshold
    }
    
    return result


def get_speaker(entry_emb, speakers_file: str = 'speakers.json', threshold: float = 0.37):
    """
    Проверяет, существует ли спикер с таким эмбеддингом в базе данных.
    
    Args:
        entry_emb: Эмбеддинг спикера для проверки (torch.Tensor)
        speakers_file: Путь к файлу speakers.json (по умолчанию 'speakers.json')
        threshold: Пороговое значение косинусного расстояния (по умолчанию 0.3)
    """
    try:
        with open(speakers_file, 'r', encoding='utf-8') as f:
            speakers_data = json.load(f)
    except FileNotFoundError:
        return "Спикер не записан", "incognito" 
    except json.JSONDecodeError:
        return "Спикер не записан", "incognito"

    if not speakers_data:
        return "Спикер не записан", "incognito"
    
    # Преобразуем entry_emb в torch.Tensor
    if not isinstance(entry_emb, torch.Tensor):
        entry_emb = torch.tensor(entry_emb, dtype=torch.float32)
    

    for speaker in speakers_data:
        speaker_emb = speaker.get('emb')
        if speaker_emb is None:
            continue
        speaker_emb_tensor = torch.tensor(speaker_emb, dtype=torch.float32)
        distance = cosine_distance(entry_emb, speaker_emb_tensor)
        if distance < threshold:
            return speaker_emb_tensor, speaker.get('name')
    return "Спикер не записан", "incognito"

