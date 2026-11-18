
import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.media.router import route_file
from src.media.config import STATIC_DIR, VISUALIZATIONS_DIR


app = FastAPI(
    title="Deepfake Detector API",
    description="Multimodal AI-powered deepfake detection API",
    version="1.0.0"
)

# Ensure static directories exist and mount static file handler
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "Deepfake Detector API is running"
    }


# Media analysis endpoint
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    suffix = os.path.splitext(file.filename)[1]

    tmp_path = None

    try:
        # Save uploaded file temporarily.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Run detection, fusion and report generation.
        result = route_file(
            tmp_path,
            original_filename=os.path.basename(file.filename)
        )

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Required model or file not found: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

    finally:
        # Always remove temporary uploaded file.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        await file.close()
