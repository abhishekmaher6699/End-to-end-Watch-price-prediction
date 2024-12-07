FROM python:3.10-slim

WORKDIR /app

COPY app_dir /app/app_dir
COPY .streamlit /app/.streamlit
COPY artifacts/data/processed/processed_data.csv /app/
COPY artifacts/models/pipeline.pkl /app/
COPY artifacts/models/columns.pkl /app/
COPY app_dir/requirements.txt /app/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

RUN ls -l /app
CMD ["streamlit", "run", "/app/app_dir/streamlit_app.py"]

