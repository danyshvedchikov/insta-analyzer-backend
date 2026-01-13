from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import base64
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SIGHTENGINE_USER = os.getenv("SIGHTENGINE_USER")
SIGHTENGINE_SECRET = os.getenv("SIGHTENGINE_SECRET")
HIVE_API_KEY = os.getenv("HIVE_API_KEY")


class AnalyzeRequest(BaseModel):
    username: str


class PhotoAnalyzeRequest(BaseModel):
    image_base64: str


@app.get("/")
def read_root():
    return {"message": "Instagram Analyzer API v3.0"}


@app.post("/analyze")
async def analyze_profile(request: AnalyzeRequest):
    username = request.username
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    
    try:
        prompt = f"""Проведи анализ Instagram профиля @{username}:
🎯 ПРОФИЛЬ: Кто это, целевая аудитория
💎 КОНТЕНТ: Темы, стиль
🌍 LIFESTYLE: Статус, интересы
📊 БИЗНЕС: Монетизация
🔮 ВЫВОД: Резюме"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        return {"analysis": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def analyze_with_sightengine(image_base64: str) -> dict:
    try:
        response = requests.post(
            'https://api.sightengine.com/1.0/check.json',
            data={
                'models': 'quality,face-attributes,genai',
                'api_user': SIGHTENGINE_USER,
                'api_secret': SIGHTENGINE_SECRET,
            },
            files={'media': ('image.jpg', base64.b64decode(image_base64 ), 'image/jpeg')}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def analyze_with_hive(image_base64: str) -> dict:
    try:
        response = requests.post(
            'https://api.thehive.ai/api/v2/task/sync',
            headers={
                'Authorization': f'Token {HIVE_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={'image': {'data': image_base64}}
         )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def format_analysis_result(sightengine_data: dict, hive_data: dict) -> str:
    result = []
    
    result.append("📸 КАЧЕСТВО ИЗОБРАЖЕНИЯ")
    if "quality" in sightengine_data:
        q = sightengine_data["quality"]
        score = int((q.get("sharpness", 0) + q.get("contrast", 0) + q.get("brightness", 0)) / 3 * 100)
        result.append(f"• Оценка: {score}/100")
        result.append(f"• Резкость: {int(q.get('sharpness', 0) * 100)}%")
        result.append(f"• Контраст: {int(q.get('contrast', 0) * 100)}%")
    
    result.append("")
    result.append("👤 ЛИЦО")
    if "faces" in sightengine_data and sightengine_data["faces"]:
        face = sightengine_data["faces"][0]
        if "attributes" in face:
            attrs = face["attributes"]
            gender = "женский" if attrs.get("female", 0) > 0.5 else "мужской"
            result.append(f"• Пол: {gender}")
            result.append(f"• Несовершеннолетний: {'да' if attrs.get('minor', 0) > 0.5 else 'нет'}")
    else:
        result.append("• Лицо не обнаружено")
    
    result.append("")
    result.append("🤖 AI-ГЕНЕРАЦИЯ")
    ai_score = 0
    if "genai" in sightengine_data:
        ai_score = sightengine_data["genai"].get("ai_generated", 0)
        result.append(f"• Sightengine: {int(ai_score * 100)}%")
    
    if "status" in hive_data and "output" in hive_data:
        for item in hive_data.get("output", []):
            for cls in item.get("classes", []):
                if "ai_generated" in cls.get("class", "").lower():
                    result.append(f"• Hive AI: {int(cls.get('score', 0) * 100)}%")
    
    result.append("")
    result.append("📊 ВЕРДИКТ")
    authenticity = int((1 - ai_score) * 100)
    result.append(f"• Аутентичность: {authenticity}/100")
    
    if authenticity > 80:
        result.append("✅ Фото выглядит натуральным")
    elif authenticity > 50:
        result.append("⚡ Есть признаки обработки")
    else:
        result.append("⚠️ Высокая вероятность AI/манипуляций")
    
    return "\n".join(result)


@app.post("/analyze-photo")
async def analyze_photo(request: PhotoAnalyzeRequest):
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="Image is required")
    
    try:
        image_data = request.image_base64
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        sightengine_result = analyze_with_sightengine(image_data)
        hive_result = analyze_with_hive(image_data)
        analysis = format_analysis_result(sightengine_result, hive_result)
        
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))







