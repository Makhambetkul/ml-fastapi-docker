# ML Model Deployment with FastAPI and Docker

A simple machine learning model served as a REST API with **FastAPI** and packaged with **Docker**.

- **Dataset:** Iris (classical dataset from scikit-learn): 150 flowers, 4 features, 3 classes.
- **Task:** multiclass classification. Predict the iris species (`setosa`, `versicolor`, `virginica`) from flower measurements.
- **Model:** `StandardScaler` + `LogisticRegression` in one scikit-learn `Pipeline`.
- **Test accuracy:** 0.933 (stratified 80/20 split, `random_state=42`).

## Project structure

```
ml-fastapi-docker/
├── main.py            # FastAPI app (GET /, POST /predict)
├── train.py           # trains the model and saves model.joblib
├── model.joblib       # saved trained model (artifact loaded by the API)
├── requirements.txt   # Python dependencies
├── Dockerfile         # container instructions
├── .dockerignore
├── .gitignore
└── README.md
```

## API endpoints

| Method | Endpoint   | Description                                   |
|--------|------------|-----------------------------------------------|
| GET    | `/`        | Checks that the API is running                |
| POST   | `/predict` | Returns the predicted species + probabilities |
| GET    | `/docs`    | Interactive Swagger documentation             |

**Example request** (`POST /predict`):

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Example response:**

```json
{
  "prediction": "setosa",
  "probabilities": {"setosa": 0.9808, "versicolor": 0.0192, "virginica": 0.0}
}
```

All features must be positive numbers (cm). Invalid input returns HTTP `422` with an explanation.

---

## 1. Run locally (Python + Uvicorn)

```bash
# create and activate a virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# train the model -> creates model.joblib
python train.py

# start the API
uvicorn main:app --reload
```

Open in the browser:

- http://127.0.0.1:8000/ → `{"message": "ML API is running"}`
- http://127.0.0.1:8000/docs → Swagger UI: open **POST /predict** → **Try it out** → **Execute**

Test with curl (macOS / Linux / Git Bash):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}'
```

## 2. Run with Docker

```bash
# build the image
docker build -t iris-api .

# run the container and expose port 8000
docker run -d -p 8000:8000 --name iris-api iris-api
```

Then test again:

- http://localhost:8000/
- http://localhost:8000/docs → test `POST /predict`

Useful commands:

```bash
docker ps                 # running containers
docker logs iris-api      # container logs
docker stop iris-api      # stop
docker rm iris-api        # remove container
```

> The Dockerfile runs `python train.py` during the build, so `model.joblib` inside the image is always created with the same scikit-learn version that the API uses. This avoids version-mismatch errors when loading the model.

## How it works

1. `train.py` trains the model and saves it to `model.joblib`. The model is an **artifact**: it is created once and then reused.
2. `main.py` loads `model.joblib` once at startup. `POST /predict` validates the input with Pydantic, converts it to a feature array in the same order as in training, and returns the prediction.
3. The **Docker image** contains the code, the dependencies and the run command, so the service runs the same way on any machine.
