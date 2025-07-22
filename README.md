# CRCXplorer 🚦

Analizador interactivo de ficheros **envioMOM\*.xml(.gz)**.

- **Trenes**: muestra el paso de trenes por circuitos TrackCircuit, con orden cronológico y gráfico interactivo.
- **Alarmas**: lista de Alarm filtrable con operadores lógicos (`AND` / `OR` / `NOT`) y salida coloreada.
- **Desvíos**: lista de Switch con posición (+ / – / movimiento) y estado de enclavamiento, también filtrable con AND/OR/NOT.
- **Señales**: lista de Signal que muestra la indicación actual de cada señal, filtrable con los mismos operadores lógicos.
- **Velocidades**: calcula el perfil de velocidad a partir de un CSV de puntos de control.

---

## Ejecución rápida con Docker

```bash
# Construir imagen (contexto actual)
docker build -t crcxplorer .

# Ejecutar (puerto 7860 en contenedor → 7860 en host)
docker run -p 7860:7860 crcxplorer
# Abre http://localhost:7860
```
> **Nota:** el contenedor instala las dependencias desde  
> `requirements-pinned.txt` (versiones probadas) o  
> `requirements-latest.txt` (versiones recientes, mismo major).  
> Cámbialo en el `Dockerfile` según prefieras estabilidad o novedades.

---

## Uso de la interfaz

1. **Subir archivos** – arrastra uno o más `envioMOM*.xml` o `envioMOM*.xml.gz`.
2. **Pestaña Trenes** – selecciona un tren de referencia para ver su recorrido y superponer el resto.
3. **Pestaña Alarmas** – filtra mensajes *Alarm* con expresiones lógicas (`CTC & ROUTE`, `!ACK`, etc.).
4. **Pestaña Desvíos** – filtra mensajes *Switch* y muestra hora, id, posición (derecha / izquierda / movimiento) y enclavado.
5. **Pestaña Señales** – filtra mensajes *Signal* y muestra hora, id e indicación (aspecto) de cada señal.
6. **Pestaña Velocidades** – sube un CSV o pégalo; se copia al cuadro para ajustarlo antes de calcular el perfil.

_Espacio → AND · `|` → OR · `!` → NOT_

---

## Instalación local (sin Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python crcxplorer.py
```

---

## Requisitos principales

```
pandas>=2.3,<3
duckdb>=1.2,<2
lxml>=5.4,<6
plotly>=5.19,<6
gradio>=5,<6
typing_extensions>=4.0
tzdata>=2024.1
```

*(La librería estándar `zoneinfo` se usa para la zona horaria.)*

---

## Licencia

MIT © 2025 Tu Nombre
