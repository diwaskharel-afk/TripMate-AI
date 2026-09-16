FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OUTPUT_DIR=/app/outputs
EXPOSE 8501

# Default: run the Streamlit demo. Override with `docker run <image> python main.py ...`
# for the CLI, or `docker run <image> python -m eval.run_eval` for the eval harness.
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
