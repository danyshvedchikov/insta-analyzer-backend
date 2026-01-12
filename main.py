import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import List

# --- КОНФИГУРАЦИЯ ---
# Ключ будет автоматически взят из переменных окружения на сервере Vercel
client = OpenAI()

app = FastAPI()

# --- Настройка CORS ---
# Разрешаем запросы от нашего будущего мобильного приложения
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для разработки разрешаем все источники
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модели данных для запросов ---

# Модель для одного поста
class Post(BaseModel):
    id: int
    caption: str | None = None # Подпись к посту
    date: str | None = None    # Дата поста

# Модель для всего запроса на анализ профиля
class ProfileAnalysisRequest(BaseModel):
    username: str
    biography: str | None = None
    external_links: List[str] | None = []
    posts: List[Post]

# --- Логика анализа ---

@app.post("/analyze-profile")
async def analyze_profile(request: ProfileAnalysisRequest):
    """
    Принимает данные профиля и постов, отправляет один большой запрос в OpenAI
    для получения структурированного аналитического отчета.
    """
    print(f"--- Получен запрос на глубокий анализ профиля: @{request.username} ---")
    print(f"Количество постов для анализа: {len(request.posts)}")

    # 1. Формируем промпт для GPT-4
    # Мы "собираем" все данные в один большой текстовый блок
    
    posts_text_block = ""
    for post in request.posts:
        posts_text_block += f"Пост #{post.id} (Дата: {post.date}):\n{post.caption}\n---\n"

    system_prompt = """
    Ты — первоклассный AI-аналитик социальных сетей и маркетолог. 
    Твоя задача — провести глубокий, структурированный анализ Instagram-профиля на основе предоставленных данных.
    Твой ответ ДОЛЖЕН быть в формате JSON. Не добавляй никаких слов до или после JSON.
    """

    user_prompt = f"""
    Проанализируй следующий профиль и его посты за последний год.

    ДАННЫЕ ПРОФИЛЯ:
    - Имя пользователя: {request.username}
    - Биография: {request.biography}
    - Внешние ссылки: {', '.join(request.external_links)}

    ДАННЫЕ ПОСТОВ:
    {posts_text_block}

    ЗАДАЧА:
    Создай детальный аналитический отчет. Включи в него следующие секции:
    1.  "general_info": Общая информация (категория блога, анализ внешних ссылок).
    2.  "content_analysis": Анализ контента (ключевые темы, тональность, динамика публикаций).
    3.  "behavioral_analysis": Поведенческий анализ (география и путешествия на основе упоминаний, стиль жизни, упоминание брендов).

    Твой ответ должен быть строго в формате JSON со следующей структурой:
    {{
      "general_info": {{
        "category": "Твоя оценка категории блога",
        "links_analysis": "Твой анализ внешних ссылок"
      }},
      "content_analysis": {{
        "main_themes": ["Тема 1", "Тема 2", "Тема 3"],
        "sentiment": "Твоя оценка тональности (позитивная, нейтральная, и т.д.)",
        "posting_dynamics": "Твой анализ частоты постов"
      }},
      "behavioral_analysis": {{
        "geography": ["Страна/Город 1", "Страна/Город 2"],
        "lifestyle": "Твой вывод о стиле жизни (например, 'Роскошь и досуг')",
        "mentioned_brands": ["Бренд 1", "Бренд 2"]
      }}
    }}
    """

    # 2. Отправляем запрос в OpenAI
    try:
        print("--- Отправляем запрос в OpenAI... ---")
        response = client.chat.completions.create(
            model="gpt-4.1-mini", # Используем быструю и умную модель
            response_format={ "type": "json_object" }, # Просим модель сразу отдать JSON
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        analysis_result = response.choices[0].message.content
        print("--- Ответ от OpenAI получен успешно! ---")
        return {"analysis": analysis_result}

    except Exception as e:
        print(f"!!! ОШИБКА ОТ OPENAI: {e} !!!")
        raise HTTPException(status_code=500, detail=f"Ошибка при обращении к OpenAI: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Insta Analyzer-Backend is running!"}
