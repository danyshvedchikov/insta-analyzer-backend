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
                            "text": """Ты — технический эксперт по цифровой обработке изображений. Проведи ТЕХНИЧЕСКИЙ анализ этого изображения на предмет цифровых манипуляций.

Оцени ТЕХНИЧЕСКИЕ аспекты:

📸 КАЧЕСТВО ИЗОБРАЖЕНИЯ: [оценка 0-100]
• Разрешение и четкость
• Артефакты сжатия
• Шумы

🎨 ЦВЕТОКОРРЕКЦИЯ:
• Применены ли фильтры (какие признаки)
• Изменение насыщенности/контраста
• Цветовой баланс

🔧 ПРИЗНАКИ РЕТУШИ:
• Сглаживание текстур
• Размытие отдельных областей
• Клонирование/удаление элементов
• Изменение пропорций

🤖 ПРИЗНАКИ AI-ГЕНЕРАЦИИ:
• Артефакты нейросетей
• Неестественные паттерны
• Аномалии в деталях

💡 ОСВЕЩЕНИЕ:
• Консистентность теней
• Направление света
• Признаки искусственного освещения

📊 ТЕХНИЧЕСКИЙ ВЕРДИКТ:
[Краткий вывод о степени обработки изображения]

Дай конкретный технический анализ этого изображения."""
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





