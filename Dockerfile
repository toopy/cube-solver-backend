FROM python:3.11

WORKDIR /app

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY ./*.py ./
COPY ./backends ./backends

ENV DEFAULT_BACKEND=dqn-solver

CMD ["fastapi", "run", "app.py"]
