FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MODEL_PATH=/app/models/best_model.pth
WORKDIR /app
# CPU-only torch keeps the image small; API needs no sklearn/matplotlib.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install fastapi "uvicorn[standard]" python-multipart pillow numpy
COPY src/__init__.py src/dataset.py src/model.py src/inference.py src/
COPY api/ api/
COPY models/best_model.pth models/best_model.pth
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
