from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/analyze")
def analyze_dummy(username: str):
    return {"message": f"Analysis for {username} will be here."}
