FROM python:3.11-slim

Instalamos dependencias del sistema (compilar lxml, etc.)

RUN apt-get update && 
apt-get install -y –no-install-recommends build-essential && 
rm -rf /var/lib/apt/lists/*

Creamos un usuario no-root para mayor seguridad

RUN useradd -m crcxplorer
WORKDIR /home/crcxplorer

Copiamos requirements y lo instalamos

COPY requirements.txt .
RUN pip install –no-cache-dir -r requirements.txt

Copiamos la aplicación

COPY crcxplorer.py .

Puerto por defecto que usa Gradio

EXPOSE 7860

Ejecutamos como usuario seguro

USER crcxplorer
CMD [“python”, “crcxplorer.py”]
