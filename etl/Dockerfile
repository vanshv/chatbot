FROM python:3.13.0-slim

WORKDIR /app

COPY ./load_graph_data.py /app
COPY ./pyproject.toml /code/pyproject.toml

RUN pip install /code/.

CMD ["python3", "load_graph_data.py"]