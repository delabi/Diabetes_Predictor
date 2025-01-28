#!/bin/bash

# Train the model and generate feature importance
python train_and_generate_importance.py

# Run the Streamlit app
streamlit run app.py
