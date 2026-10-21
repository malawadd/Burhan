FROM python:3.13-slim
WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && useradd --uid 10001 --create-home burhan
COPY backend/*.py /app/burhan/backend/
COPY assets/catalogue.json /app/burhan/assets/catalogue.json
COPY assets/research-catalogue.json /app/burhan/assets/research-catalogue.json
RUN mkdir -p /var/lib/burhan && chown -R burhan:burhan /var/lib/burhan
USER burhan
ENV HOST=0.0.0.0 PORT=3210 SMART_INSPECTION_PROVIDER=aws BURHAN_DATA_DIR=/var/lib/burhan AWS_REGION=us-east-1
EXPOSE 3210
CMD ["python", "-m", "burhan.backend"]
