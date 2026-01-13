from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AnalyzeRequest(BaseModel):
    username: str


class PhotoAnalyzeRequest(BaseModel):
    image_base64: str


@app.get("/")
def read_root():
    return {"message": "Instagram Analyzer API v2.0"}


@app.post("/analyze")
async def analyze_profile(request: AnalyzeRequest):
    username = request.username
    
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    
    try:
        prompt = f"""Ты — эксперт-аналитик Instagram профилей. Проведи глубокий анализ профиля @{username}.

Структура анализа:

🎯 ПРОФИЛЬ И ПОЗИЦИОНИРОВАНИЕ
• Кто этот человек/бренд
• Целевая аудитория
• Уникальное позиционирование

💎 КОНТЕНТ-СТРАТЕГИЯ
• Основные темы контента
• Стиль подачи
• Частота публикаций

🌍 LIFESTYLE
• География и путешествия
• Маркеры статуса
• Ценности и интересы

👗 СТИЛЬ И ЭСТЕТИКА
• Визуальный стиль
• Модные предпочтения
• Цветовая палитра

📊 БИЗНЕС-ПОТЕНЦИАЛ
• Возможности монетизации
• Потенциальные коллаборации

🔮 ИТОГОВАЯ ГИПОТЕЗА
Краткое резюме: кто этот человек и какова его стратегия в Instagram.

Будь конкретным и избегай общих фраз."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты — профессиональный аналитик социальных сетей."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        analysis = response.choices[0].message.content
        return {"analysis": analysis}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-photo")
async def analyze_photo(request: PhotoAnalyzeRequest):
    
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="Image is required")
    
    try:
        image_data = request.image_base64
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Проанализируй это фото и дай КОНКРЕТНЫЙ результат:

📸 АУТЕНТИЧНОСТЬ: [число]/100

🎭 ФИЛЬТРЫ И ОБРАБОТКА:
• [Конкретно что обнаружено или "Не обнаружено"]

🤖 AI-ГЕНЕРАЦИЯ: [X]% вероятность
• [Признаки почему]

👤 ЧЕЛОВЕК НА ФОТО:
• Возраст на фото: ~[X] лет
• Реальный возраст (оценка): ~[X] лет
• Косметические процедуры: [да/нет, какие]

🔍 ЧТО ИЗМЕНЕНО:
• Кожа: [что именно]
• Лицо: [что именно]
• Фон: [что именно]

📊 ВЕРДИКТ:
[2-3 предложения: насколько фото реальное и что изменено]

ВАЖНО: Дай КОНКРЕТНЫЕ ответы на основе этого фото, не общие советы!"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        analysis = response.choices[0].message.content
        return {"analysis": analysis}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


