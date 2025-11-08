import json
import time
import threading
from pathlib import Path
from datetime import datetime
import torchaudio
from tts import tts_function

EVENTS_FILE = "events.json"
RESULTS_FILE = "results.json"
OUTPUT_DIR = Path("output")
CHECK_INTERVAL = 60  # секунд (1 минута)

# Создаем папку output если её нет
OUTPUT_DIR.mkdir(exist_ok=True)


def get_audio_duration(wav_path):
    """
    Получает длительность аудио файла в секундах.
    """
    try:
        info = torchaudio.info(wav_path)
        duration = info.num_frames / info.sample_rate
        return duration
    except Exception as e:
        print(f"Ошибка при получении длительности аудио: {e}")
        return 0.0


def add_result_to_json(wav_path, text, speaker_name="Unknown"):
    """
    Добавляет запись в results.json с автоматической генерацией id.
    
    Args:
        wav_path: Путь до wav файла
        text: Текст, поданный в TTS
        speaker_name: Имя спикера (заглушка)
        
    Returns:
        id созданной записи
    """
    # Читаем существующий файл или создаем пустой список
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                results_data = json.loads(content)
                if not isinstance(results_data, list):
                    results_data = []
            else:
                results_data = []
    except (FileNotFoundError, json.JSONDecodeError):
        results_data = []
    
    # Определяем id: если список пустой, то 1, иначе максимальный id + 1
    if not results_data:
        result_id = 1
    else:
        # Находим максимальный id
        max_id = max(result.get('id', 0) for result in results_data if isinstance(result, dict))
        result_id = max_id + 1
    
    # Получаем длительность аудио
    duration = get_audio_duration(wav_path)
    
    # Получаем текущее время
    current_time = datetime.now().isoformat()
    
    # Создаем новую запись
    new_result = {
        "id": result_id,
        "wav_path": str(wav_path),
        "speaker_name": speaker_name,
        "duration": duration,
        "time": current_time,
        "text": text
    }
    
    # Добавляем новую запись
    results_data.append(new_result)
    
    # Записываем обратно в файл
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    return result_id


def process_events():
    """
    Читает events.json, находит записи с is_gen="False",
    вызывает tts_function для каждой записи и удаляет запись после успешной генерации.
    """
    try:
        # Читаем файл
        with open(EVENTS_FILE, 'r', encoding='utf-8') as f:
            events = json.load(f)
        
        # Находим записи, которые нужно обработать
        events_to_process = [event for event in events if event.get("is_gen") == "False"]
        
        if not events_to_process:
            print("Нет записей для обработки")
            return
        
        print(f"Найдено {len(events_to_process)} записей для обработки")
        
        # Список id_event для удаления после успешной обработки
        processed_event_ids = []
        
        # Обрабатываем каждую запись
        for event in events_to_process:
            event_id = event.get("id_event")
            text = event.get("text", "")
            emb = event.get("emb", [])
            cond = event.get("cond", [])
            
            if not text or not emb or not cond:
                print(f"Пропущена запись {event_id}: отсутствует text, emb или cond")
                continue
            
            print(f"Обработка записи {event_id}...")
            
            # Генерируем путь к выходному файлу в папке output
            out_wav = OUTPUT_DIR / f"output_{event_id}.wav"
            
            try:
                # Вызываем функцию генерации аудио
                tts_function(text=text, emb=emb, cond=cond, out_wav=str(out_wav))
                
                # Добавляем запись в results.json
                result_id = add_result_to_json(
                    wav_path=out_wav,
                    text=text,
                    speaker_name="Unknown"  # Заглушка
                )
                
                # Добавляем id_event в список для удаления после успешной генерации
                processed_event_ids.append(event_id)
                print(f"Запись {event_id} успешно обработана, добавлена в results.json с id={result_id}")
                
            except Exception as e:
                print(f"Ошибка при обработке записи {event_id}: {e}")
                # Не удаляем запись при ошибке, чтобы можно было повторить попытку
        
        # Удаляем успешно обработанные записи из events
        if processed_event_ids:
            events = [event for event in events if event.get("id_event") not in processed_event_ids]
            print(f"Удалено {len(processed_event_ids)} записей из events.json")
        
        # Сохраняем обновленный файл
        with open(EVENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        
        print("Файл events.json обновлен")
        
    except FileNotFoundError:
        print(f"Файл {EVENTS_FILE} не найден")
    except json.JSONDecodeError as e:
        print(f"Ошибка при чтении JSON: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


def run_periodic_check():
    """
    Запускает периодическую проверку файла events.json каждую минуту.
    """
    print("Запуск периодической проверки events.json...")
    print(f"Интервал проверки: {CHECK_INTERVAL} секунд")
    
    while True:
        try:
            process_events()
        except Exception as e:
            print(f"Ошибка в процессе проверки: {e}")
        
        # Ждем перед следующей проверкой
        time.sleep(CHECK_INTERVAL)


def start_background_check():
    """
    Запускает проверку в фоновом потоке.
    """
    thread = threading.Thread(target=run_periodic_check, daemon=True)
    thread.start()
    print("Фоновая проверка запущена")
    return thread


if __name__ == "__main__":
    # Запускаем проверку
    # Можно запустить в фоне или в основном потоке
    print("Запуск обработки событий...")
    run_periodic_check()

