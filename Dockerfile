# API service: serves the best model exported by train.py (MLflow)
FROM python:3.12-slim

WORKDIR /app

# Install API dependencies (MLflow is not needed to serve the model)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the exported model + its MLflow info
COPY main.py model.joblib model_info.json ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
