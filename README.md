# CRCXplorer 🚦

Analizador interactivo de ficheros **envioMOM\*.xml(.gz)**.

- **Trenes**: muestra el paso de trenes por circuitos TrackCircuit, con orden cronológico y gráfico interactivo.
- **Alarmas**: lista de Alarm filtrable con operador lógico simple (`AND` / `OR` / `NOT`) y salida coloreada.

---

## Ejecución rápida con Docker

```bash
# Construir imagen (contexto actual)
docker build -t crcxplorer .

# Ejecutar (puerto 7860 en contenedor → 7860 en host)
docker run -p 7860:7860 crcxplorer
# Abre http://localhost:7860
```

---

## Uso de la interfaz

1. **Subir archivos** – arrastra uno o más `envioMOM*.xml` o `envioMOM*.xml.gz`.
2. **Pestaña Trenes** – selecciona un tren de referencia.  
   - El eje Y sigue exactamente el orden de circuitos de ese tren.  
   - Los demás trenes aparecen superpuestos.
3. **Pestaña Alarmas** – escribe filtros como:

   * `CTC & ROUTE`
   * `MANDOS | RBC`
   * `!ACK`

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
pandas>=2.0
duckdb>=0.9
lxml>=5.0
plotly>=5.16
gradio>=4.0
```

*(La librería estándar `zoneinfo` se usa para la zona horaria.)*

---

## Captura de pantalla

![Pantalla CRCXplorer](assets/screenshot.png)

---

## Licencia

MIT © 2025 Tu Nombre
