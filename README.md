# 🔄 Natural Language to SQL Converter
> Transform Natural Language into SQL queries with the power of LLMs! 🚀

This project showcases a complete pipeline for converting natural language queries into SQL using a fine-tuned T5-Large model with LoRA. Built with simplicity and efficiency in mind, it features a user-friendly Streamlit interface for real-time query conversion.

## ✨ Project Overview

🤖 **Model Fine-Tuning**
- T5-Large model fine-tuned on a custom Text-to-SQL dataset
- LoRA implementation for efficient parameter adaptation
- Optimized for accuracy and performance

📊 **Key Components**
- **Data Generation:** Synthetic training data creation via `data_generation.py`
- **Helper Utilities:** Robust preprocessing and logging in `helper.py`
- **Interactive UI:** Sleek Streamlit interface in `app.py`
- **Comprehensive Logging:** Detailed logs for debugging and monitoring
- **Checkpoint System:** Ready for resume capability implementation

## 📁 Repository Structure
```bash
Natural_Language-to-SQL/
├── app.py                 # Streamlit web application
├── requirements.txt       # Project dependencies
├── final_model           # Trained model directory
├── src/
│   ├── __init__.py
│   ├── helper.py         # Utility functions
├── CUDA_check.py         # GPU compatibility check
├── Finetuning.py        # Model training script
├── data_generation.py    # Training data generator
├── nl_sql_dataset.csv    # Dataset file
├── setup.py             # Project setup script
```

## 🌟 Features

- 💡 Intuitive natural language processing
- ⚡ Fast and efficient query conversion
- 🎯 High accuracy with T5-Large model
- 📱 User-friendly web interface
- 📊 Comprehensive logging system

## 🚀 Quick Start Guide

```bash
git clone https://github.com/PrathamAhuja1/Natural_Language-to-SQL.git
cd Natural_Language-to-SQL
pip install -r requirements.txt
streamlit run app.py
```
