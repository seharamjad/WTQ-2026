import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()  # .env file se GEMINI_API_KEY padhta hai

st.set_page_config(page_title="Feedback Helper", page_icon="📊")
st.title("📊 Customer Feedback Helper")

client = genai.Client()  # GEMINI_API_KEY environment se khud leta hai

# Agar ye model na chale to "gemini-3.1-flash-lite" try karein (dono allowed hain)
MODEL = "gemini-3.5-flash-lite"


def ask_ai(prompt: str) -> str:
    try:
        reply = client.models.generate_content(model=MODEL, contents=prompt)
        return reply.text
    except Exception as e:
        return f"Kuch masla aaya: {e}"


def classify_reviews(texts):
    """Har review ke liye Positive/Negative/Neutral ki list wapas deta hai."""
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        "Har review ko 'Positive', 'Negative' ya 'Neutral' mein classify karein. "
        "Sirf ek JSON list wapas dein, review ki tarteeb mein, "
        'jaise ["Positive", "Negative"]. Koi aur text na likhein.\n\n' + numbered
    )
    raw = ask_ai(prompt).replace("```json", "").replace("```", "").strip()
    labels = json.loads(raw)
    if len(labels) != len(texts):
        raise ValueError("AI ne galat tadaad mein jawab diya, dobara try karein.")
    return labels


# 1) Data
file = st.file_uploader("Apni CSV upload karein (column ka naam: review)", type="csv")
df = pd.read_csv(file) if file else pd.read_csv("sample_data.csv")

if "review" not in df.columns:
    st.error("CSV mein 'review' naam ka column hona chahiye.")
    st.stop()

df = df.dropna(subset=["review"]).head(50).reset_index(drop=True)
reviews = "\n".join(f"- {text}" for text in df["review"])

st.subheader("Data")
st.dataframe(df, use_container_width=True)

# 2) Summary
st.subheader("AI Summary")
if st.button("Summary banao"):
    with st.spinner("AI soch raha hai..."):
        st.write(
            ask_ai(
                "Neeche customer reviews hain. 5 bullet points mein batayein ke "
                "customers kis cheez ki tareef karte hain aur kis par shikayat.\n\n"
                + reviews
            )
        )

# 3) Sentiment + chart (NAYA)
st.subheader("Sentiment Analysis")
if st.button("Sentiment nikalo"):
    with st.spinner("Reviews classify ho rahe hain..."):
        try:
            df["sentiment"] = classify_reviews(df["review"].tolist())
            st.session_state["result"] = df
        except Exception as e:
            st.error(f"Classification nahi ho saki: {e}")

if "result" in st.session_state:
    result = st.session_state["result"]
    st.bar_chart(result["sentiment"].value_counts())
    st.dataframe(result, use_container_width=True)

# 4) Sawal-jawab
st.subheader("Data se sawal poochein")
question = st.text_input("Misal: Sab se zyada shikayat kis cheez par hai?")
if question:
    with st.spinner("AI jawab dhoond raha hai..."):
        st.write(
            ask_ai(
                "Sirf in reviews ki bunyaad par jawab dein. Agar jawab data mein "
                "na ho to kahein 'data mein nahi hai'.\n\n"
                f"Reviews:\n{reviews}\n\nSawal: {question}"
            )
        )
