#!/usr/bin/env python
# crcxplorer v1.0  (Python ≥ 3.9)

from __future__ import annotations
import gzip, io, inspect, struct
from pathlib import Path
from typing import Union, List, Tuple, Dict
from datetime import datetime
import duckdb, pandas as pd
import plotly.express as px
import gradio as gr
from lxml import etree
from datetime import timezone
from zoneinfo import ZoneInfo
import re
# ──────────────────────────────────────────────────────────────────────────────
# Filtrado booleano simple  AND / OR / NOT  (subcadenas, sin regex)
# ──────────────────────────────────────────────────────────────────────────────
def _build_mask(df: pd.DataFrame, expr: str, col: str = "raw") -> pd.Series:
    """
    Expr sencilla:
      AND → espacio o "&"
      OR  → "|"
      NOT → "!" delante
    Sin paréntesis; evalúa de izquierda a derecha.
    """
    expr = expr.strip()
    if not expr:
        return pd.Series(True, index=df.index)
    or_parts = [p.strip() for p in re.split(r"\|", expr)]
    mask_total = pd.Series(False, index=df.index)
    for part in or_parts:
        if not part:
            continue
        and_terms = [t for t in re.split(r"&|\s+", part) if t]
        mask_and = pd.Series(True, index=df.index)
        for term in and_terms:
            neg = term.startswith("!")
            term_clean = term[1:] if neg else term
            term_match = df[col].str.contains(term_clean, case=False, regex=False)
            mask_and &= ~term_match if neg else term_match
        mask_total |= mask_and
    return mask_total
# Colores para Alarmas
_ALARM_COLORS = ["#e45a1a", "#377eb8", "#4daf4a", "#984ea3"]

def _colorize_alarm(row) -> str:
    """
    Devuelve una línea HTML con columnas alineadas y coloreadas:
    [fecha‑hora] [event] [mensaje] [descripción]
    """
    # Campos
    ts   = row["time"].strftime("%Y-%m-%d %H:%M:%S")   # incluye fecha
    ev   = row["event"]
    desc = row["descr"]
    msg  = row["msg"]

    # Anchuras fijas
    widths = [20, 15, 25, 30]     # ajusta si necesitas
    parts  = [ts, ev, desc, msg]

    def _pad(text, w):
        txt = (text[:w]) if len(text) > w else text.ljust(w)
        return txt.replace(" ", "&nbsp;")               # conservar alineación en HTML

    html_parts = []
    for i, (p, w) in enumerate(zip(parts, widths)):
        html_parts.append(
            f"<span style='color:{_ALARM_COLORS[i%4]};font-family:monospace'>{_pad(p, w)}</span>"
        )
    return "".join(html_parts)

# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE FICHERO  (acepta .gz ó plano)
# ──────────────────────────────────────────────────────────────────────────────
FileLike = Union[str, Path, bytes, io.BufferedIOBase, dict]

def _open_maybe_gz(path: Path, mode: str = "rb"):
    return gzip.open(path, mode) if path.suffix == ".gz" else open(path, mode)

def _read_bytes(file_obj: FileLike) -> bytes:
    """Lee bytes del archivo admitiendo:
    • objetos bytes / bytearray
    • dict FileData de Gradio   {"data": …, "path": …}
    • rutas str / Path (xml o xml.gz)
    • file‑like ya abierto

    Si los primeros dos bytes son 1f 8b (cabecera gzip) se descomprime al vuelo.
    """
    # ── 1) bytes o bytearray ───────────────────────────────────────
    if isinstance(file_obj, (bytes, bytearray)):
        data = bytes(file_obj)
        return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data

    # ── 2) dict FileData (Gradio) ─────────────────────────────────
    if isinstance(file_obj, dict):
        if file_obj.get("data"):
            data = file_obj["data"]
            return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        # Si solo hay ruta
        file_obj = file_obj.get("path") or file_obj.get("name")

    # ── 3) ruta en disco (str/Path) ───────────────────────────────
    if isinstance(file_obj, (str, Path)):
        path = Path(file_obj)
        with _open_maybe_gz(path) as fh:
            return fh.read()

    # ── 4) file‑like abierto ──────────────────────────────────────
    data = file_obj.read()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data

def _strip_to_xml(data: bytes) -> bytes:
    """
    Devuelve 'data' a partir del primer '<', quitando BOM UTF-8 y/o
    cualquier carácter basura previo. Lanza ValueError si no hay '<'.
    """
    # Eliminar BOM UTF-8
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    idx = data.find(b"<")
    if idx == -1:
        raise ValueError("No se encontró inicio de XML '<'")
    return data[idx:]

# ──────────────────────────────────────────────────────────────────────────────
# PARSER XML - TrackCircuit
# ──────────────────────────────────────────────────────────────────────────────
NS = {"x": "http://www.w3.org/2001/XMLSchema-instance"}  # namespace abreviado

def _iter_trackcircuit(xml_bytes: bytes):
    """Yield (timestamp, tc_id, train_id) de cada mensaje TrackCircuit con train."""
    xml_bytes = _strip_to_xml(xml_bytes)   # asegura que empieza en '<'
    # Los ficheros MOM contienen muchos <Message> consecutivos sin un nodo raíz.
    # Envolvemos el contenido para que sea XML bien formado.
    xml_bytes = b"<root>" + xml_bytes + b"</root>"
    ctx = etree.iterparse(
        io.BytesIO(xml_bytes),
        events=("end",),
        tag="Message",
        recover=True,
        huge_tree=True,
    )
    for _event, elem in ctx:
        try:
            ctype = elem.findtext("Header/ContentType")
            if ctype != "TrackCircuit":
                elem.clear(); continue

            # idTrain en TrainOccupation ó en Trains/Train
            train_occ = elem.find(".//TrainOccupation", namespaces=NS)
            train_id  = train_occ.get("idTrain") if train_occ is not None else ""

            if not train_id:
                elem.clear(); continue

            tc_id = elem.findtext(".//TrackCircuit/Id", namespaces=NS) or "UNKNOWN"

            # Hora REAL del evento (TrackCircuit/TimeStampValue) en ms UTC
            ts_attr = elem.find(".//TrackCircuit/TimeStampValue")
            ts_raw = int(ts_attr.get("value"))
            # Pásala a la zona deseada (UTC+2; cambio automático verano/invierno)
            ts = datetime.fromtimestamp(ts_raw / 1000.0, tz=timezone.utc).astimezone(ZoneInfo("Europe/Madrid"))

            yield ts, tc_id, train_id
        finally:
            elem.clear()

# ──────────────────────────────────────────────────────────────────────────────
# PARSER XML - Alarm
# ──────────────────────────────────────────────────────────────────────────────
def _iter_alarm(xml_bytes: bytes):
    """
    Yield (ts, event, msg, descr, raw) de cada mensaje Alarm con contenido.
    """
    xml_bytes = _strip_to_xml(xml_bytes)
    xml_bytes = b"<root>" + xml_bytes + b"</root>"
    ctx = etree.iterparse(
        io.BytesIO(xml_bytes),
        events=("end",),
        tag="Message",
        recover=True,
        huge_tree=True,
    )
    for _e, elem in ctx:
        try:
            if elem.findtext("Header/ContentType") != "Alarm":
                elem.clear(); continue
            al = elem.find(".//Alarm")
            if al is None:
                elem.clear(); continue
            ts_raw = int(al.find(".//Timestamp").get("value"))
            ts = datetime.fromtimestamp(ts_raw / 1000.0, tz=timezone.utc).astimezone(
                ZoneInfo("Europe/Madrid")
            )
            ev = al.find(".//EventType").get("value")
            prm_el = al.find(".//Parameters/Parameter")
            msg = prm_el.get("value") if prm_el is not None else ""
            descr = al.findtext(".//Description") or ""
            raw = f"{ev} {msg} {descr}"
            yield ts, ev, msg, descr, raw
        finally:
            elem.clear()

def _parse_files(file_objs: List[FileLike]) -> pd.DataFrame:
    """Devuelve un DataFrame con columnas: time, train, tcid."""
    rows: List[Tuple[datetime, str, str]] = []
    for fo in file_objs:
        raw = _read_bytes(fo)
        xml_bytes = _strip_to_xml(raw)
        rows.extend(_iter_trackcircuit(xml_bytes))
    df = pd.DataFrame(rows, columns=["time", "tcid", "train"])
    df.sort_values("time", inplace=True)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE NEGOCIO
# ──────────────────────────────────────────────────────────────────────────────
def _trenes_de_referencia(df: pd.DataFrame) -> List[str]:
    return sorted(df["train"].unique())

def _circuitos_de_tren(df: pd.DataFrame, train: str) -> List[str]:
    """Orden de paso (primera vez)"""
    sel = (
        df[df["train"] == train]          # solo el tren elegido
          .sort_values("time")            # orden temporal estricto
          .drop_duplicates("tcid")        # primera vez en cada circuito
    )
    return sel["tcid"].tolist()

def _plot(df: pd.DataFrame, ref_circs: List[str]) -> px.line:
    """Devuelve un gráfico de paso de trenes ajustado en altura,
    sin título y con la leyenda abajo."""
    if not ref_circs:
        raise ValueError("Seleccione un tren de referencia.")

    # Filtrar y fijar orden de circuitos
    df2 = df[df["tcid"].isin(ref_circs)].copy()
    df2["tcid"] = pd.Categorical(df2["tcid"], categories=ref_circs, ordered=True)
    

    fig = px.line(
        df2,
        x="time",
        y="tcid",
        color="train",
        markers=True,
        labels={"time": "Hora", "tcid": "Circuito", "train": "Tren"},
    )

    # ── eje-Y exactamente en el orden del tren de referencia ──
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=ref_circs,
    )

    # Altura dinámica (mín 300 px, 25 px por circuito)
    fig.update_layout(
        height=max(300, len(ref_circs) * 25),
        title=None,
        legend=dict(
            orientation="h",
            y=1.08,     # arriba del área del gráfico
            x=0,
            xanchor="left",
            font=dict(size=10),
        ),
        margin=dict(t=55, b=30, l=60, r=20),
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# CALLBACKS GRADIO
# ──────────────────────────────────────────────────────────────────────────────
def cb_alarmas(files, filtro_txt):
    if not files:
        return "⚠️ Suba un fichero."
    rows = []
    for fo in (files if isinstance(files, list) else [files]):
        raw = _read_bytes(fo)
        rows.extend(_iter_alarm(raw))
    if not rows:
        return "⚠️ No se encontraron alarmas."
    df = pd.DataFrame(rows, columns=["time", "event", "msg", "descr", "raw"])
    mask = _build_mask(df, filtro_txt or "", col="raw")
    df = df[mask]
    if df.empty:
        return "⚠️ Ninguna alarma coincide con el filtro."
    html_lines = [_colorize_alarm(r) for _, r in df.iterrows()]
    return "<br>".join(html_lines)
def cb_update_trenes(files):
    if not files: return gr.update(choices=[], value=None)
    df = _parse_files(files if isinstance(files, list) else [files])
    return gr.update(choices=_trenes_de_referencia(df), value=None)

def cb_plot(files, tren_ref):
    if not files: return "⚠️ Suba un fichero.", None
    df = _parse_files(files if isinstance(files, list) else [files])
    if not tren_ref: return "⚠️ Seleccione un tren.", None
    ref_c = _circuitos_de_tren(df, tren_ref)
    if not ref_c: return f"⚠️ El tren {tren_ref} no pasa por circuitos válidos.", None
    fig = _plot(df, ref_c)
    return "", fig

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="CRC Explorer") as demo:
    gr.Markdown("# CRCXplorer – Análisis de ficheros envioMOM")
    files = gr.Files(label="Ficheros xml o xml.gz", file_count="multiple",
                     file_types=[".xml", ".gz"])

    with gr.Tabs():
        with gr.TabItem("Trenes"):
            tren_sel = gr.Dropdown(label="Tren de referencia (muestra circuitos)", choices=[])
            files.change(cb_update_trenes, inputs=files, outputs=tren_sel)

            btn = gr.Button("Generar gráfico")
            out_msg = gr.Markdown()
            out_plot = gr.Plot()
            btn.click(cb_plot, inputs=[files, tren_sel], outputs=[out_msg, out_plot])

        with gr.TabItem("Alarmas"):
            filtro_alarm = gr.Text(label="Filtro (AND/&  OR/|  NOT/!)")
            btn_alarm = gr.Button("Filtrar alarmas")
            out_alarm = gr.HTML()
            btn_alarm.click(
                cb_alarmas,
                inputs=[files, filtro_alarm],
                outputs=out_alarm,
            )

    gr.Markdown("---\nSube uno o varios `envioMOM*.xml(.gz)` y explora. Las pestañas:\n"
                "• **Trenes** – recorrido de circuitos\n"
                "• **Alarmas** – filtra mensajes Alarm con AND/OR/NOT.")
    
if __name__ == "__main__":
    # Importante para Docker
    demo.launch(server_name="0.0.0.0", server_port=7860)
