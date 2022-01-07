FROM python:3.9
WORKDIR /Volcanic-eruption-prediction
COPY . /Volcanic-eruption-prediction
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
EXPOSE 5000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "5000"]