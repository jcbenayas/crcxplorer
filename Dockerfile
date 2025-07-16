# ────────────────────────────────────────────────────────────────────────────────
# Imagen ligera basada en Python 3.11
FROM python:3.11-slim

# 1. Instalar dependencias del sistema necesarias para matplotlib / fontconfig
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        fontconfig \
        && rm -rf /var/lib/apt/lists/*

# 2. Crear usuario no-root para ejecutar la app
RUN useradd -m crcxplorer
WORKDIR /home/crcxplorer
# Matplotlib necesita un directorio de caché escribible
ENV MPLCONFIGDIR=/home/crcxplorer/.cache/matplotlib
RUN mkdir -p $MPLCONFIGDIR && chown -R crcxplorer:crcxplorer $MPLCONFIGDIR

# 3. Copiar código y dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY crcxplorer.py .

# 4. Puerto por defecto de Gradio
EXPOSE 7862

# 5. Comando de arranque
USER crcxplorer
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "crcxplorer.py"]
