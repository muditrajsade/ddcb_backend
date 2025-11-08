from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
import os
from openai import OpenAI
from google.cloud import aiplatform

# --- ENV CONFIG ---
os.environ["GOOGLE_CLOUD_PROJECT"] = "ham10000-477009"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"  # update this if needed
os.environ["OPENAI_API_KEY"] = "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"

# --- INIT ---
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
app = FastAPI()

# Optional: Vertex AI setup (if you want to use your trained HAM10000 model)
PROJECT_ID = "18409880463"
REGION = "us-central1"
ENDPOINT_ID = "2693828776818638848"
aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(ENDPOINT_ID)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev, you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL MAP ---
HAM10000_MAP = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "nv": "Melanocytic nevi (common moles)",
    "mel": "Melanoma (skin cancer)",
    "vasc": "Vascular lesions (angiomas, angiokeratomas, etc.)",
}


# --- ROUTES ---

@app.get("/")
def home():
    return {"message": "Dermatology AI Backend is running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), description: str = Form("")):
    """
    1. Receives an image and optional description
    2. Uses Vertex AI for skin classification
    3. Uses OpenAI to generate friendly summary
    """
    # Read image
    image_bytes = await file.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Predict via Vertex AI model
    response = endpoint.predict(instances=[{"content": encoded_image}])
    pred = response.predictions[0]
    confidences = np.array(pred["confidences"])
    labels = pred["displayNames"]
    top_idx = np.argmax(confidences)
    top_label = labels[top_idx]
    top_conf = confidences[top_idx]
    disease_name = HAM10000_MAP.get(top_label, top_label)

    # Generate friendly explanation with OpenAI
    prompt = f"""
    The image was classified as {disease_name} with {top_conf*100:.2f}% confidence.
    The user described: {description}

    Write a friendly, educational summary for a dermatology assistant including:
    - Overview
    - Causes
    - Treatment options (mention drugs and dosages if relevant)
    - Self-care advice
    - Disclaimer that this is not medical diagnosis.
    """

    llm_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a kind dermatology assistant providing safe, educational responses."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=800,
    )

    summary = llm_response.choices[0].message.content.strip()

    return {
        "condition": disease_name,
        "confidence": round(float(top_conf), 4),
        "summary": summary,
    }


class DrugInfoRequest(BaseModel):
    drug_name: str
@app.post("/drug-info")
async def get_drug_info(req: DrugInfoRequest):
    """
    Fetch concise drug information from OpenFDA API.
    """
    base_url = "https://api.fda.gov/drug/label.json"
    params = {"search": f"openfda.brand_name:{req.drug_name}", "limit": 1}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            return {"error": f"No information found for {req.drug_name}"}

        info = data["results"][0]
        raw_text = str(info)

        # Summarize with OpenAI for readable output
        summary_prompt = f"""
        Summarize this FDA drug label info for {req.drug_name} into 5 short points:
        - Usage
        - Dosage
        - Warnings
        - Common side effects
        - Contraindications
        Text: {raw_text[:4000]}
        """

        llm_summary = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You summarize FDA drug label info for users clearly and safely."},
                {"role": "user", "content": summary_prompt},
            ],
            temperature=0.5,
            max_tokens=600,
        )

        summary = llm_summary.choices[0].message.content.strip()
        return {"drug": req.drug_name, "summary": summary}

    except Exception as e:
        return {"error": f"Failed to fetch data for {req.drug_name}: {e}"}

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Continues chat conversation with OpenAI.
    """
    llm_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly dermatology assistant."},
            {"role": "user", "content": req.message},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    reply = llm_response.choices[0].message.content.strip()
    return {"reply": reply}

