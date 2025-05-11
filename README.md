# 🤖 Prompt Refinement Coach

A Streamlit web app that transforms vague, informal instructions into **clear, agent-ready prompts**. Built as a capstone project after completing the Coursera specialization **“Agentic AI and AI Agents for Leaders”** by [Dr. Jules White](https://www.linkedin.com/in/juleswhite/).

This tool helps professionals communicate more effectively with AI agents like GPT-4, ReAct, and Operator models—by turning intent into structured action.

---

## 🔍 Live Demo
👉 [Launch the app on Streamlit Cloud](https://h2qtrtv3kqcbhjnbxhggmh.streamlit.app/)

---

## 🎥 Walkthrough Video
[Watch the demo](https://youtu.be/35Pv9dm9Xl0)

---

## 📚 Course Inspiration

This project was inspired by what I learned in:
- Agentic AI fundamentals
- Custom GPT creation and deployment
- Advanced prompt engineering techniques

---

## 🛠️ Features

- Accepts vague user input (e.g., "run something for our product")
- Outputs structured, step-by-step prompts
- Dropdown to tailor for domains (Business, HR, Marketing)
- Sample prompt button
- Built using HuggingFace Zephyr-7B via inference API

---

## 📦 Installation

```bash
# Create virtual environment
conda create -n prompt-coach python=3.10
conda activate prompt-coach

# Install dependencies
pip install -r requirements.txt
