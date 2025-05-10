import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

EXAMPLES = {
    "Business": "Send the latest report to the manager.",
    "HR": "Schedule a check-in with the new hire.",
    "Marketing": "Plan a campaign for the new product launch.",
}

def call_huggingface(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"⚠️ Error: {response.status_code} - {response.text}"

def construct_system_prompt(domain, user_input):
    return (
        f"Refine the following vague instruction into a clear, structured prompt for an AI agent "
        f"in the {domain} domain using best practices (clarity, step-by-step format, positive phrasing):\n\n"
        f"---\n"
        f"Original: {user_input.strip()}\n"
        f"---\n"
        f"Improved: (use clear, numbered steps like 1., 2., 3.):"
    )

# Streamlit UI
st.set_page_config(page_title="Prompt Refinement Coach", layout="centered")
st.title("🤖 Prompt Refinement Coach")
st.write("Transform vague instructions into clear, structured prompts for AI agents.")

domain = st.selectbox("Select a prompt domain:", ["Business", "HR", "Marketing", "General"])

default_input = st.text_area("Enter your vague instruction:", height=120)

if st.button("Use Sample Prompt"):
    default_input = EXAMPLES.get(domain, list(EXAMPLES.values())[0])
    st.session_state["input"] = default_input
    st.experimental_rerun()

if st.button("Refine Prompt"):
    if not default_input.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Refining..."):
            formatted_prompt = construct_system_prompt(domain, default_input)
            output = call_huggingface(formatted_prompt)
            st.markdown("### 🔍 Refined Prompt")
            st.success(output.replace(formatted_prompt, "").strip())
