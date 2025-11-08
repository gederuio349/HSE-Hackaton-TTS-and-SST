from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import os
import tempfile
from pathlib import Path
import torch
import numpy as np
from stt import stt_function
import logging
from pydub import AudioSegment
import io
import soundfile as sf
import librosa
import json
import base64
from vector import get_speaker

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="STT API", description="API для обработки аудио файлов")


def check_json():
     with open('speakers.json', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Проверяем, пуст ли файл
            if not content:
                print('json пустой')
                return True


            # Пытаемся загрузить JSON
            speakers_data = json.loads(content)
            # Проверяем, есть ли записи
            if not speakers_data:
                print('json пустой')
                return True
   
            
            # Если это список, проверяем, не пустой ли он
            elif isinstance(speakers_data, list) and len(speakers_data) == 0:
                print('json пустой')
                return True
 
            
            # Если это словарь, проверяем, не пустой ли он
            elif isinstance(speakers_data, dict) and len(speakers_data) == 0:
                print('json пустой')
                return True

            else:
                return False

            
# Добавляем CORS middleware для работы с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаем директорию для временных файлов
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def convert_tensors_to_dict(latents):
    """
    Конвертирует torch тензоры в numpy массивы для JSON сериализации
    """
    converted = {}
    for key, value in latents.items():
        if isinstance(value, torch.Tensor):
            # Конвертируем в numpy и затем в список
            converted[key] = value.cpu().detach().numpy().tolist()
        else:
            converted[key] = value
    return converted


def add_speaker_to_json(speaker_name, speaker_embedding, gpt_cond_latent):
    """
    Добавляет нового спикера в speakers.json с автоматической генерацией id
    
    Args:
        speaker_name: Имя спикера
        speaker_embedding: Эмбеддинг спикера
        gpt_cond_latent: GPT условный латент
        
    Returns:
        id созданного спикера
    """
    # Читаем существующий файл или создаем пустой список
    try:
        with open('speakers.json', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                speakers_data = json.loads(content)
                if not isinstance(speakers_data, list):
                    speakers_data = []
            else:
                speakers_data = []
    except (FileNotFoundError, json.JSONDecodeError):
        speakers_data = []
    
    # Определяем id: если список пустой, то 1, иначе максимальный id + 1
    if not speakers_data:
        speaker_id = 1
    else:
        # Находим максимальный id
        max_id = max(speaker.get('id', 0) for speaker in speakers_data if isinstance(speaker, dict))
        speaker_id = max_id + 1
    
    # Создаем новую запись спикера
    new_speaker = {
        "id": speaker_id,
        "name": speaker_name,
        "emb": speaker_embedding,
        "cond": gpt_cond_latent,
    }
    
    # Добавляем новую запись
    speakers_data.append(new_speaker)
    
    # Записываем обратно в файл
    with open('speakers.json', 'w', encoding='utf-8') as f:
        json.dump(speakers_data, f, ensure_ascii=False, indent=2)
    
    return speaker_id


def add_event_to_json(text, speaker_embedding, gpt_cond_latent):
    """
    Добавляет новое событие в events.json с автоматической генерацией id_event
    
    Args:
        text: Текст события
        speaker_embedding: Эмбеддинг спикера
        gpt_cond_latent: GPT условный латент
        
    Returns:
        id_event созданного события
    """
    # Читаем существующий файл или создаем пустой список
    try:
        with open('events.json', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                events_data = json.loads(content)
                if not isinstance(events_data, list):
                    events_data = []
            else:
                events_data = []
    except (FileNotFoundError, json.JSONDecodeError):
        events_data = []
    
    # Определяем id_event: если список пустой, то 1, иначе максимальный id + 1
    if not events_data:
        id_event = 1
    else:
        # Находим максимальный id_event
        max_id = max(event.get('id_event', 0) for event in events_data if isinstance(event, dict))
        id_event = max_id + 1
    
    # Создаем новое событие
    new_event = {
        "id_event": id_event,
        "text": text,
        "emb": speaker_embedding,
        "cond": gpt_cond_latent,
        "is_gen": "False"
    }
    
    # Добавляем новую запись
    events_data.append(new_event)
    
    # Записываем обратно в файл
    with open('events.json', 'w', encoding='utf-8') as f:
        json.dump(events_data, f, ensure_ascii=False, indent=2)
    
    return id_event


# Обработчик ошибок валидации для диагностики
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Обработчик ошибок валидации - помогает увидеть детали ошибки 422
    """
    logger.error(f"Ошибка валидации: {exc.errors()}")
    logger.error(f"URL: {request.url}")
    logger.error(f"Метод: {request.method}")
    logger.error(f"Заголовки: {dict(request.headers)}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Ошибка валидации запроса. Убедитесь, что файл отправляется как multipart/form-data с ключом 'audio'",
            "hint": "Используйте FormData на фронтенде: const formData = new FormData(); formData.append('audio', audioBlob, fileName);"
        }
    )


@app.post("/stt")
async def process_audio(audio: UploadFile = File(...), speaker_name: str = Form(...)):
    """
    Принимает аудио файл, сохраняет его и обрабатывает через stt_function
    
    Args:
        audio: Загруженный аудио файл (должен быть отправлен как multipart/form-data с ключом 'audio')
        
    Returns:
        JSON с транскрибированным текстом и эмбеддингами
    """
    logger.info(f"Получен запрос на обработку файла: {audio.filename if audio else 'None'}")
    logger.info(f"Content-Type: {audio.content_type if audio else 'None'}")
    print(speaker_name)
    # Проверяем формат файла
    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail="Файл не предоставлен")
    
    # Создаем временный файл
    file_extension = Path(audio.filename).suffix.lower()
    if file_extension not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
        raise HTTPException(
            status_code=400, 
            detail=f"Неподдерживаемый формат файла. Поддерживаются: .wav, .mp3, .ogg, .flac, .m4a"
        )
    
    try:
        # Читаем содержимое файла
        content = await audio.read()
        
        # Сохраняем оригинальный файл
        temp_file_path = UPLOAD_DIR / f"temp_{audio.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)
        
        # Конвертируем файл в валидный WAV формат
        # Пробуем разные методы конвертации
        wav_file_path = UPLOAD_DIR / f"converted_{Path(audio.filename).stem}.wav"
        final_file_path = temp_file_path
        
        # Метод 1: Пытаемся через pydub
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(content))
            audio_segment.export(str(wav_file_path), format="wav")
            final_file_path = wav_file_path
        except Exception as conv_error:
            try:
                # Загружаем аудио через librosa из сохраненного файла (автоматически определяет формат)
                audio_data, sample_rate = librosa.load(str(temp_file_path), sr=None)
                # Сохраняем как WAV через soundfile
                sf.write(str(wav_file_path), audio_data, sample_rate)
                final_file_path = wav_file_path
            except Exception as librosa_error:
                pass

        # Вызываем stt_function
        text, latents = stt_function(str(final_file_path))
        # Конвертируем тензоры в формат, пригодный для JSON

        latents_dict = convert_tensors_to_dict(latents)

        is_speakers_empty = check_json()

        if speaker_name != 'incognito':
            # Создаем новую запись для speakers.json
            add_speaker_to_json(
                speaker_name=speaker_name,
                speaker_embedding=latents_dict['speaker_embedding'],
                gpt_cond_latent=latents_dict['gpt_cond_latent']
            )
            is_speakers_empty = False
        

        ##################
        # поиск спикера по косиносному расстоянию
        if (not is_speakers_empty) and (speaker_name == 'incognito'):
            orig_speaker_tensor = get_speaker(latents_dict['speaker_embedding'])
            if orig_speaker_tensor != "Спикер не записан":
                latents_dict['speaker_embedding'] = orig_speaker_tensor.cpu().detach().numpy().tolist()
            else:
                return JSONResponse(content={
                    "is_speakers_empty": "True",
                    "success": "True"
                    })
           
        ####################

        

        if temp_file_path.exists():
            os.remove(temp_file_path)
        if final_file_path != temp_file_path and final_file_path.exists():
            os.remove(final_file_path)
        
        if  is_speakers_empty:
            return JSONResponse(content={
            "is_speakers_empty": "True",
            "success": "True"
        })


        # Создаём новую запись в event
        add_event_to_json(
            text=text,
            speaker_embedding=latents_dict['speaker_embedding'],
            gpt_cond_latent=latents_dict['gpt_cond_latent']
        )

        return JSONResponse(content={
            "is_speakers_empty": "False",
            "success": "True"
        })
        
    except Exception as e:
        print("ошибка", e)
        logger.error(f"Ошибка при обработке файла: {str(e)}", exc_info=True)
        # Удаляем временные файлы в случае ошибки
        temp_file_path = UPLOAD_DIR / f"temp_{audio.filename}"
        if temp_file_path.exists():
            os.remove(temp_file_path)
        
        # Удаляем конвертированный файл, если он был создан
        wav_file_path = UPLOAD_DIR / f"converted_{Path(audio.filename).stem}.wav"
        if wav_file_path.exists():
            os.remove(wav_file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Корневой эндпоинт для проверки работы API
    """
    return {"message": "STT API работает", "endpoints": ["/stt", "/messages", "/health", "/docs"]}


@app.get("/health")
async def health_check():
    """
    Проверка здоровья API
    """
    return {"status": "healthy"}


@app.get("/messages")
async def get_messages():
    """
    Получает все сообщения из results.json и возвращает их с аудио файлами в формате base64
    
    Returns:
        JSON массив с сообщениями, где каждый элемент содержит:
        - id: идентификатор сообщения
        - speaker_name: имя спикера
        - audio_data: аудио файл в формате base64 (data URI)
        - duration: длительность аудио
        - timestamp: время создания
        - text: транскрибированный текст
    """
    try:
        # Читаем results.json
        results_path = Path("results.json")
        if not results_path.exists():
            logger.warning("Файл results.json не найден, возвращаем пустой список")
            return []
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        if not isinstance(results_data, list):
            logger.warning("results.json не содержит список, возвращаем пустой список")
            return []
        
        # Обрабатываем каждое сообщение
        messages = []
        for item in results_data:
            try:
                # Получаем путь к аудио файлу
                wav_path = item.get('wav_path', '')
                if not wav_path:
                    logger.warning(f"Пропускаем сообщение {item.get('id', 'unknown')}: нет пути к аудио файлу")
                    continue
                
                # Преобразуем путь (может быть с обратными слешами на Windows)
                audio_file_path = Path(wav_path)
                
                # Проверяем существование файла
                if not audio_file_path.exists():
                    logger.warning(f"Аудио файл не найден: {audio_file_path}")
                    # Продолжаем, но без аудио данных
                    audio_data_uri = None
                else:
                    # Читаем аудио файл и конвертируем в base64
                    with open(audio_file_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        # Создаем data URI для аудио
                        audio_data_uri = f"data:audio/wav;base64,{audio_base64}"
                
                # Формируем сообщение в формате, ожидаемом фронтендом
                message = {
                    "id": item.get('id'),
                    "file_name": f"output_{item.get('id', 'unknown')}.wav",  # Для совместимости
                    "speaker_name": item.get('speaker_name', 'Unknown'),
                    "user_name": item.get('speaker_name', 'Unknown'),  # Альтернативное поле
                    "userName": item.get('speaker_name', 'Unknown'),  # Еще одно альтернативное поле
                    "audio_data": audio_data_uri,  # Base64 данные
                    "audioUrl": audio_data_uri,  # Для совместимости с фронтендом
                    "duration": item.get('duration', 0),
                    "time": item.get('time', ''),
                    "timestamp": item.get('time', ''),
                    "created_at": item.get('time', ''),
                    "text": item.get('text', ''),
                    "transcription": item.get('text', '')  # Альтернативное поле
                }
                
                messages.append(message)
                
            except Exception as e:
                logger.error(f"Ошибка при обработке сообщения {item.get('id', 'unknown')}: {str(e)}", exc_info=True)
                # Продолжаем обработку других сообщений
                continue
        
        return messages
        
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка при чтении results.json: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении results.json: {str(e)}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении сообщений: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при получении сообщений: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4010)

