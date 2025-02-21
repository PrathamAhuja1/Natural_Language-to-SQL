# Natural Language to SQL Converter

This project demonstrates a full pipeline for converting natural language queries into SQL queries using a fine-tuned T5-Large model with LoRA. It includes scripts for model fine-tuning, synthetic data generation, and a Streamlit web UI for real-time inference.

## Project Overview

- **Model Fine-Tuning:** The T5-Large model is fine-tuned on a custom Text-to-SQL dataset using LoRA for efficient parameter adaptation.
- **Data Generation:** The `data_generation.py` script generates synthetic training data in CSV format.
- **Helper Utilities:** The `helper.py` file contains utility functions for data preprocessing, logging, and prompt formatting.
- **Streamlit UI:** The `app.py` file provides an interactive web interface where users can input a natural language query and obtain the corresponding SQL query.
- **Logging:** All logs are stored in the `logs/` folder for easy debugging and monitoring.
- **Resume Capability:** Checkpoint saving and resuming functionality can be added as needed.

## Repository Structure

```bash
Natural_Language-to-SQL/
├── app.py
├── requirements.txt
├── final_model
├── src/
│   ├── __init__.py
│   ├── helper.py
├── CUDA_check.py
│──Finetuning.py
│──data_generation.py
├──nl_sql_dataset.csv
├──setup.py
└── templates/
    └── chat.html
```
