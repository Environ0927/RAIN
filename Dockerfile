FROM python:3.11-slim
WORKDIR /artifact
COPY . .
RUN python -m pip install --no-cache-dir -e ".[test]"
CMD ["python", "-m", "pytest", "-q"]
