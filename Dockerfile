FROM python:3.9-slim

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

ENV NAME World

CMD ["python", "main.py"]
