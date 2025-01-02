from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React static files in production
app.mount("/", StaticFiles(directory="../ui/dist", html=True))

@app.post("/api/predict")
async def predict(file: UploadFile):
    # Here you'll integrate with your ML model
    return {"breed": "predicted_breed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
