from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained Aqua AI model
model = joblib.load("aqua_ai_model.pkl")

app = FastAPI(title="Aqua AI ML API")

# Allow frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# WATER SENSOR DATA
# =========================================================

class WaterReading(BaseModel):
    pH: float
    TDS: float
    Turbidity: float
    Temperature: float


# =========================================================
# AQUA AI CHAT REQUEST
# =========================================================

class ChatRequest(BaseModel):
    message: str
    risk: str
    ph: float
    tds: float
    turbidity: float
    temperature: float
    wqi: float


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return {
        "message": "Aqua AI ML API is running",
        "model": "Random Forest"
    }


# =========================================================
# WATER QUALITY PREDICTION
# =========================================================

@app.post("/predict")
def predict(reading: WaterReading):

    data = pd.DataFrame([{
        "pH": reading.pH,
        "TDS": reading.TDS,
        "Turbidity": reading.Turbidity,
        "Temperature": reading.Temperature
    }])

    # Get prediction
    prediction = model.predict(data)[0]

    # Get probability for each risk class
    probabilities = model.predict_proba(data)[0]

    probability_result = {
        class_name: round(float(probability), 4)
        for class_name, probability
        in zip(model.classes_, probabilities)
    }

    return {
        "risk": prediction,
        "probabilities": probability_result
    }


# =========================================================
# AQUA AI CHATBOT
# =========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.lower().strip()

    risk = str(request.risk).upper()

    ph = request.ph
    tds = request.tds
    turbidity = request.turbidity
    temperature = request.temperature
    wqi = request.wqi


    # =====================================================
    # FULL WATER QUALITY SUMMARY
    # =====================================================

    if (
        "explain" in message
        or "summary" in message
        or "summarize" in message
        or "current readings" in message
        or "all readings" in message
        or "overall" in message
        or "water quality report" in message
    ):

        response = (
            f"<strong>Current Water Quality Summary</strong><br><br>"
            f"• pH: {ph}<br>"
            f"• TDS: {tds} ppm<br>"
            f"• Turbidity: {turbidity} NTU<br>"
            f"• Temperature: {temperature} °C<br>"
            f"• WQI: {wqi}/100<br>"
            f"• Predicted contamination risk: <strong>{risk}</strong><br><br>"
            f"Based on the current sensor readings, Aqua AI predicts a "
            f"<strong>{risk.lower()} contamination risk</strong>."
        )


    # =====================================================
    # WHY IS THE RISK HIGH?
    # =====================================================

    elif (
        "why" in message and "risk" in message
        or "why is the risk" in message
        or "why high" in message
        or "reason for high" in message
        or "what caused" in message
    ):

        concerns = []

        if ph < 6.5 or ph > 8.5:
            concerns.append(f"pH ({ph})")

        if tds > 500:
            concerns.append(f"TDS ({tds} ppm)")

        if turbidity > 5:
            concerns.append(f"turbidity ({turbidity} NTU)")

        if concerns:

            response = (
                f"The current predicted risk is <strong>{risk}</strong>.<br><br>"
                f"Parameters that may be contributing to the current "
                f"condition include:<br>"
                + "<br>".join(f"• {item}" for item in concerns)
            )

        else:

            response = (
                f"The current predicted risk is <strong>{risk}</strong>. "
                "The individual readings do not show an obvious abnormal "
                "parameter based on the simple monitoring thresholds used "
                "by this assistant."
            )


    # =====================================================
    # IS THE WATER OKAY?
    # =====================================================

    elif (
        "is the water okay" in message
        or "is water okay" in message
        or "is the water safe" in message
        or "can i drink" in message
        or "drinkable" in message
        or "good water" in message
    ):

        response = (
            f"Aqua AI currently predicts a "
            f"<strong>{risk.lower()} contamination risk</strong>.<br><br>"
            f"The current WQI displayed by the dashboard is "
            f"<strong>{wqi}/100</strong>.<br><br>"
            "This dashboard prediction is intended for monitoring and "
            "early-warning purposes and should not by itself be treated "
            "as confirmation that water is safe for drinking."
        )


    # =====================================================
    # WHICH PARAMETER IS ABNORMAL?
    # =====================================================

    elif (
        "which parameter" in message
        or "abnormal parameter" in message
        or "abnormal" in message
        or "problem with" in message
        or "what is wrong" in message
    ):

        abnormal = []

        if ph < 6.5 or ph > 8.5:
            abnormal.append(f"pH: {ph}")

        if tds > 500:
            abnormal.append(f"TDS: {tds} ppm")

        if turbidity > 5:
            abnormal.append(f"Turbidity: {turbidity} NTU")

        if not abnormal:

            response = (
                "No obvious abnormal parameter was detected by the "
                "simple threshold checks used by Aqua AI."
            )

        else:

            response = (
                "<strong>Parameters requiring attention:</strong><br><br>"
                + "<br>".join(f"• {item}" for item in abnormal)
            )


    # =====================================================
    # RISK
    # =====================================================

    elif (
        "risk" in message
        or "contamination" in message
        or "contamination level" in message
    ):

        response = (
            f"The current predicted contamination risk is "
            f"<strong>{risk}</strong>.<br><br>"
            f"The latest readings are pH {ph}, TDS {tds} ppm, "
            f"turbidity {turbidity} NTU, and temperature "
            f"{temperature} °C."
        )


    # =====================================================
    # pH
    # =====================================================

    elif "ph" in message:

        response = (
            f"The current water pH is <strong>{ph}</strong>.<br><br>"
            "pH indicates how acidic or alkaline the water is. "
            "A value closer to neutral is generally preferred "
            "for water quality monitoring."
        )


    # =====================================================
    # TDS
    # =====================================================

    elif (
        "tds" in message
        or "total dissolved solids" in message
        or "dissolved solids" in message
    ):

        response = (
            f"The current TDS level is <strong>{tds} ppm</strong>.<br><br>"
            "TDS represents the concentration of dissolved substances "
            "in the water."
        )


    # =====================================================
    # TURBIDITY
    # =====================================================

    elif (
        "turbidity" in message
        or "cloudy" in message
        or "cloudiness" in message
    ):

        response = (
            f"The current turbidity level is "
            f"<strong>{turbidity} NTU</strong>.<br><br>"
            "Turbidity indicates the amount of suspended particles "
            "that can make water appear cloudy."
        )


    # =====================================================
    # TEMPERATURE
    # =====================================================

    elif (
        "temperature" in message
        or "temp" in message
        or "hot" in message
        or "cold" in message
    ):

        response = (
            f"The current water temperature is "
            f"<strong>{temperature} °C</strong>."
        )


    # =====================================================
    # WQI
    # =====================================================

    elif (
        "wqi" in message
        or "water quality index" in message
        or "water quality score" in message
    ):

        response = (
            f"The current Water Quality Index displayed by Aqua AI "
            f"is <strong>{wqi}/100</strong>.<br><br>"
            f"The current predicted contamination risk is "
            f"<strong>{risk}</strong>."
        )


    # =====================================================
    # HELP
    # =====================================================

    elif (
        "help" in message
        or "what can you do" in message
        or "what can i ask" in message
    ):

        response = (
            "<strong>I can help you monitor the water quality.</strong><br><br>"
            "You can ask me about:<br>"
            "• Current contamination risk<br>"
            "• pH<br>"
            "• TDS<br>"
            "• Turbidity<br>"
            "• Temperature<br>"
            "• WQI<br>"
            "• Current readings<br>"
            "• Abnormal parameters<br>"
            "• Why the risk is high"
        )


    # =====================================================
    # GREETING
    # =====================================================

    elif (
        message in ["hi", "hello", "hey", "hii", "hey aqua ai"]
        or message.startswith("hello ")
        or message.startswith("hi ")
    ):

        response = (
            "Hello! 👋 I’m Aqua AI, your water quality monitoring "
            "assistant. Ask me about the current sensor readings, "
            "WQI, or contamination risk."
        )


    # =====================================================
    # GENERAL QUESTION
    # =====================================================

    else:

        response = (
            f"I’m Aqua AI, your water quality monitoring assistant. "
            f"The current predicted contamination risk is "
            f"<strong>{risk}</strong>.<br><br>"
            "Try asking:<br>"
            "• Explain the current readings<br>"
            "• What is the current pH?<br>"
            "• What is the TDS?<br>"
            "• Which parameter is abnormal?<br>"
            "• Why is the risk high?<br>"
            "• What is the WQI?"
        )


    return {
        "response": response
    }