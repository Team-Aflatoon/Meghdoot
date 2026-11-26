# Local replica of the judge container (R1_CONTRACT.md): CPU torch + the listed libraries, nothing else.
# Build once (needs internet):  docker build -f scripts/judge.Dockerfile -t monsoon-judge scripts
# Run offline:                  bash scripts/judge_check.sh
FROM python:3.12-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy scipy pandas xarray scikit-learn lightgbm onnxruntime fastapi flask uvicorn gunicorn requests
