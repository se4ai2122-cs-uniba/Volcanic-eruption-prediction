FROM python:3.9
COPY test.zip test.zip
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt 
COPY locustfile.py locustfile.py
EXPOSE 8089
CMD ["locust", "-f", "locustfile.py", "--host", "http://host.docker.internal:5000"]