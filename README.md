CRCXplorer 🚦

Analizador interactivo de ficheros envioMOM*.xml(.gz).
	•	Trenes: muestra el paso de trenes por circuitos TrackCircuit, con orden cronológico y gráfico interactivo.
	•	Alarmas: lista de Alarm filtrable con operador lógico simple (AND / OR / NOT) y salida coloreada.

⸻

Ejecución rápida con Docker

# Construir imagen (contexto actual)
docker build -t crcxplorer .

# Ejecutar (puerta de enlace en el puerto 7860)
docker run -p 7860:7860 crcxplorer
# → abre http://localhost:7860


⸻

Uso de la interfaz
	1.	Subir archivos – arrastra uno o más envioMOM*.xml o envioMOM*.xml.gz.
	2.	Pestaña Trenes – selecciona un tren de referencia.
	•	El eje-Y se ordena tal como ese tren recorre los circuitos.
	•	Los demás trenes aparecen superpuestos.
	3.	Pestaña Alarmas – escribe filtros como:
	•	CTC & ROUTE
	•	MANDOS | RBC
	•	!ACK
Espacio » AND · | » OR · ! » NOT

⸻

Instalación local (sin Docker)

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python crcxplorer.py


⸻

Requisitos principales

pandas>=2.0
duckdb>=0.9
lxml>=5.0
plotly>=5.16
gradio>=4.0

(La librería estándar zoneinfo se usa para la zona horaria.)

⸻

Captura de pantalla


⸻

Licencia

MIT © 2025 Juan Carlos Benayas
