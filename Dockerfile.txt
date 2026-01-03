FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "Web_Interface_View.py", "--server.port=8080", "--server.address=0.0.0.0"]
