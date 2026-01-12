import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# --- КОНФИГУРАЦИЯ ---
# Вставляем API ключ прямо в код для надежности

API_KEY = "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"

client = OpenAI(api_key=API_KEY)


app = FastAPI()

# --- Настройка CORS ---
# Разрешаем запросы от нашего расширения
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешает все источники, для разработки это нормально
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модели данных для запросов ---
# Определяем, какие данные мы ожидаем от расширения
class PostData(BaseModel):
    image_base64: str
    caption: str | None = None # Текст поста, может быть пустым

# --- ЭНДПОИНТ ДЛЯ АНАЛИЗА ОДНОГО ПОСТА ---
@app.post("/analyze-post")
async def analyze_post(post_data: PostData):
    print("--- Шаг 1: Получен запрос на анализ поста (картинка + текст) ---")
    
    # Убираем заголовок "data:image/jpeg;base64," из строки, если он есть
    if "," in post_data.image_base64:
        base64_image = post_data.image_base64.split(',')[1]
    else:
        base64_image = post_data.image_base64

    # --- УЛУЧШЕННЫЙ ПРОМПТ ДЛЯ AI ---
    prompt_text = f"""
    Ты — AI-аналитик Instagram-профилей. Проанализируй следующий пост.
    Текст поста: "{post_data.caption if post_data.caption else 'Текст отсутствует.'}"

    На основе текста и изображения, кратко, в одном-двух предложениях, опиши суть поста. 
    Определи, является ли это рекламой, личным контентом, экспертным мнением или чем-то еще.
    """

    try:
        print("--- Шаг 2: Отправляем запрос в OpenAI... ---")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        
        analysis_result = response.choices[0].message.content
        print(f"--- Шаг 3: Получен ответ от OpenAI: {analysis_result[:100]}... ---")
        
        # Возвращаем результат анализа обратно в расширение
        return {"summary": analysis_result}

    except Exception as e:
        print(f"!!! ОШИБКА ОТ OPENAI: {e} !!!")
        # Если что-то пошло не так, сообщаем об этом расширению
        raise HTTPException(status_code=500, detail=f"Ошибка при обращении к OpenAI: {str(e)}")

# --- Главная страница для проверки работы сервера ---
@app.get("/")
def read_root():
    return {"message": "Сервер InstaAnalyzer запущен и готов к работе!"}