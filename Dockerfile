# 1. Python base image
FROM python:3.12-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Install dependencies first (this layer is cached if requirements.txt does not change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy project files into the container
COPY . .

# 5. Re-train the model inside the image, so model.joblib is created
#    with exactly the same scikit-learn version that the API uses
RUN python train.py

# 6. Port of the API
EXPOSE 8000

# 7. Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
