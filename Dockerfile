FROM python:3.7.9-buster

WORKDIR /app
EXPOSE 8000
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY data /app/data
COPY src /app
RUN python -m stocks train
CMD ["python", "-m", "stocks", "server", "-p", "8000"]
