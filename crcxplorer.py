#!/usr/bin/env python
# crcxplorer v1.0  (Python ≥ 3.9)

from __future__ import annotations
import gzip, io, inspect, struct
from pathlib import Path
from typing import Union, List, Tuple, Dict
from datetime import datetime, timedelta
import duckdb, pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from lxml import etree
from datetime import timezone
from zoneinfo import ZoneInfo
import re
import os

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

# ──────────────────────────────────────────────────────────────────────────────
# Helper: convertir epoch‑milliseconds → datetime con µs exactos
# Evita la pérdida de precisión por float
# ──────────────────────────────────────────────────────────────────────────────
def _dt_from_ms(ms_val: int, tz=timezone.utc, local_tz: str = "Europe/Madrid") -> datetime:
    """
    Convierte milisegundos desde época Unix a datetime con microsegundos,
    sin pasar por float. Devuelve en la zona horaria indicada (`local_tz`).
    """
    secs, millis = divmod(int(ms_val), 1_000)
    dt = datetime.fromtimestamp(secs, tz=tz) + timedelta(milliseconds=millis)
    return dt.astimezone(ZoneInfo(local_tz)) if local_tz else dt

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

def _colorize_switch(row) -> str:
    """Devuelve HTML monoespaciado: [hora.ms] [id] [posición] [enclavado]"""
    ts  = row["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    sid = row["id"]
    pos = row["posicion"]
    enc = row["enclavado"]

    widths = [30, 25, 20, 15]              # más separación
    parts  = [ts, sid, pos, enc]
    colors = ["#ffa500", "#1e90ff", "#00ced1", "#adff2f"]

    def pad(t, w):
        txt = (t[:w]) if len(t) > w else t.ljust(w)
        return txt.replace(" ", "&nbsp;")

    html = [
        f"<span style='font-family:monospace;color:{colors[i]}'>{pad(p, w)}</span>"
        for i, (p, w) in enumerate(zip(parts, widths))
    ]
    return "".join(html)

# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE FICHERO  (acepta .gz ó plano)
# ──────────────────────────────────────────────────────────────────────────────
FileLike = Union[str, Path, bytes, io.BufferedIOBase, dict]

def _read_bytes(file_obj: FileLike) -> bytes:
    """Lee bytes del archivo admitiendo:
    • bytes / bytearray
    • dict FileData de Gradio  {"data":…, "path":…}
    • rutas str / Path (cualquier nombre, .gz o no)
    • file-like ya abierto

    Si los dos primeros bytes son 1F 8B (cabecera gzip) se descomprime.
    """
    # ── 1) bytes o bytearray ──────────────────────────────────────
    if isinstance(file_obj, (bytes, bytearray)):
        data = bytes(file_obj)
        return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data

    # ── 2) dict FileData (Gradio) ────────────────────────────────
    if isinstance(file_obj, dict):
        if file_obj.get("data"):
            data = file_obj["data"]
            return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        file_obj = file_obj.get("path") or file_obj.get("name")

    # ── 3) ruta en disco (str / Path) ────────────────────────────
    if isinstance(file_obj, (str, Path)):
        path = Path(file_obj)
        with path.open("rb") as fh:
            data = fh.read()
            return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data

    # ── 4) file-like abierto ─────────────────────────────────────
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
            ts = _dt_from_ms(ts_raw)
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
            ts = _dt_from_ms(ts_raw)
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
# PARSER XML – Switch  (desvíos)
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# PARSER XML – Switch  (desvíos)
# ──────────────────────────────────────────────────────────────────────────────
def _iter_switch(xml_bytes: bytes):
    """
    Yield → (ts, posicion, enclavado, raw)
        ts         : datetime con milisegundos
        posicion   : 'derecha' | 'izquierda' | 'movimiento'
        enclavado  : 'enclavado' | 'no enclavado'
        raw        : texto simple para filtrado
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
    for _ev, elem in ctx:
        try:
            if elem.findtext("Header/ContentType") != "Switch":
                elem.clear(); continue

            sw = elem.find(".//Switch")
            if sw is None:
                elem.clear(); continue

            # ── timestamp ───────────────────────────────────────────────
            ts_raw = int(elem.findtext("Header/Timestamp"))
            ts = _dt_from_ms(ts_raw)

            # ── nombre del desvío / Id --────────────────────────────────
            switch_id = sw.findtext("Id") or "UNKNOWN"

            # ── posición ────────────────────────────────────────────────
            chk = sw.find(".//CheckedPosition")
            val, pos_attr = chk.get("value"), chk.get("position")
            if val == "YES" and pos_attr == "+":
                posicion = "derecha"
            elif val == "YES" and pos_attr == "-":
                posicion = "izquierda"
            else:
                posicion = "movimiento"

            # ── enclavado ───────────────────────────────────────────────
            enc_el = sw.find(".//Interlocked")
            enclavado = "enclavado" if enc_el is not None and enc_el.get("value") == "YES" else "no enclavado"

            raw = f"{switch_id} {posicion} {enclavado}"
            yield ts, switch_id, posicion, enclavado, raw
        finally:
            elem.clear()

# ──────────────────────────────────────────────────────────────────────────────
# PARSER XML – Signal  (señales)
# ──────────────────────────────────────────────────────────────────────────────
def _iter_signal(xml_bytes: bytes):
    """
    Yield → (ts, id, indicacion, raw)
        ts          : datetime con milisegundos
        id          : nombre de la señal (ej. LAV6.TRO.1795S)
        indicacion  : texto de <Indication>
        raw         : texto simple para filtrado
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
    for _ev, elem in ctx:
        try:
            if elem.findtext("Header/ContentType") != "Signal":
                elem.clear(); continue

            sig = elem.find(".//Signal")
            if sig is None:
                elem.clear(); continue

            ts_raw = int(elem.findtext("Header/Timestamp"))
            ts = _dt_from_ms(ts_raw)

            signal_id  = sig.findtext("Id") or "UNKNOWN"
            indicacion = sig.findtext(".//Indication") or ""

            raw = f"{signal_id} {indicacion}"
            yield ts, signal_id, indicacion, raw
        finally:
            elem.clear()

# Colores para Señales
def _colorize_signal(row) -> str:
    """Devuelve HTML monoespaciado: [hora.ms] [id] [indicación]"""
    ts  = row["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    sid = row["id"]
    ind = row["indicacion"]

    widths = [30, 25, 40]
    parts  = [ts, sid, ind]
    colors = ["#ffa500", "#1e90ff", "#4caf50"]

    def pad(t, w):
        txt = (t[:w]) if len(t) > w else t.ljust(w)
        return txt.replace(" ", "&nbsp;")

    html = [
        f"<span style='font-family:monospace;color:{colors[i]}'>{pad(p, w)}</span>"
        for i, (p, w) in enumerate(zip(parts, widths))
    ]
    return "".join(html)

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
    # Formato eje‑X con ms (Plotly %L = milliseconds)
    fig.update_xaxes(tickformat="%H:%M:%S.%L")

    # ── eje‑Y exactamente en el orden del tren de referencia ──
    n_cats = len(ref_circs)
    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=ref_circs,
        dtick=1,                    # paso uniforme
        range=[n_cats - 1, 0],      # sin ½‐paso adicional → todas las filas igual
    )

    # ── distribución y leyenda ────────────────────────────────────────────
    fig.update_layout(
        height=max(300, len(ref_circs) * 25),
        title=None,
        legend=dict(
            orientation="v",
            y=0.5,  yanchor="middle",
            x=1.05, xanchor="left",   # un poco más fuera
            font=dict(size=10),
        ),
        margin=dict(t=50, b=30, l=60, r=200),
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

def cb_switch(files, filtro_txt):
    """Filtrado booleano (AND/&  OR/|  NOT/!) para desvíos."""
    if not files:
        return "⚠️ Suba un fichero."
    rows = []
    for fo in (files if isinstance(files, list) else [files]):
        raw = _read_bytes(fo)
        rows.extend(_iter_switch(raw))
    if not rows:
        return "⚠️ No se encontraron desvíos."

    df = pd.DataFrame(rows, columns=["time", "id", "posicion", "enclavado", "raw"])
    mask = _build_mask(df, filtro_txt or "", col="raw")
    df = df[mask]
    if df.empty:
        return "⚠️ Ningún desvío coincide con el filtro."

    html_lines = [_colorize_switch(r) for _, r in df.iterrows()]
    return "<br>".join(html_lines)

# Callback para señales
def cb_signal(files, filtro_txt):
    """Filtrado booleano (AND/&  OR/|  NOT/!) para señales."""
    if not files:
        return "⚠️ Suba un fichero."
    rows = []
    for fo in (files if isinstance(files, list) else [files]):
        raw = _read_bytes(fo)
        rows.extend(_iter_signal(raw))
    if not rows:
        return "⚠️ No se encontraron señales."

    df = pd.DataFrame(rows, columns=["time", "id", "indicacion", "raw"])
    mask = _build_mask(df, filtro_txt or "", col="raw")
    df = df[mask]
    if df.empty:
        return "⚠️ Ninguna señal coincide con el filtro."

    html_lines = [_colorize_signal(r) for _, r in df.iterrows()]
    return "<br>".join(html_lines)

def cb_update_trenes(files):
    if not files: return gr.update(choices=[], value=None)
    df = _parse_files(files if isinstance(files, list) else [files])
    return gr.update(choices=_trenes_de_referencia(df), value=None)

def cb_update_circuitos(files, tren_ref):
    """
    Devuelve la lista de circuitos (tcid) recorridos por el tren seleccionado,
    todos marcados por defecto. Si no hay tren, lista vacía.
    """
    if not files or not tren_ref:
        return gr.update(choices=[], value=[])
    df = _parse_files(files if isinstance(files, list) else [files])
    circs = _circuitos_de_tren(df, tren_ref)
    return gr.update(choices=circs, value=circs)

def cb_plot(files, tren_ref, circs_sel):
    """
    Genera la gráfica de trenes filtrando por los circuitos seleccionados.
    Si el usuario no selecciona ninguno, se usan todos los del tren de referencia.
    """
    if not files:
        return "⚠️ Suba un fichero.", None
    df = _parse_files(files if isinstance(files, list) else [files])
    if not tren_ref:
        return "⚠️ Seleccione un tren.", None

    # Todos los circuitos por los que pasa el tren (orden temporal 1ª vez)
    all_circs = _circuitos_de_tren(df, tren_ref)
    if not all_circs:
        return f"⚠️ El tren {tren_ref} no pasa por circuitos válidos.", None

    # Usar selección del usuario si existe; si no, todos
    if not circs_sel:
        ref_c = all_circs
    else:
        sel_set = set(circs_sel)
        ref_c = [c for c in all_circs if c in sel_set]

    if not ref_c:
        return "⚠️ Seleccione al menos un punto de control.", None

    fig = _plot(df, ref_c)
    return "", fig


# ──────────────────────────────────────────────────────────────────────────────
# CALLBACK Velocidades
# ──────────────────────────────────────────────────────────────────────────────
def cb_velocidades(files, csv_text, csv_file, dir_mad):
    if not files:
        return "⚠️ Suba al menos un fichero MOM.", None, None, None

    # ── leer puntos (tcid, distancia en metros) ──────────────────────────────
    puntos_df = None
    try:
        if csv_text:
            import io
            puntos_df = pd.read_csv(io.StringIO(csv_text),
                                    header=None, names=["tcid", "dist"])
        elif csv_file:
            puntos_df = pd.read_csv(csv_file.name, header=None,
                                    names=["tcid", "dist"])
    except Exception as e:
        return f"⚠️ Error leyendo CSV de puntos: {e}", None, None, None

    if puntos_df is None or puntos_df.empty:
        return "⚠️ Introduzca o suba un CSV con puntos (tcid,dist).", None, None, None

    # Distancia a numérica y ordenar
    puntos_df["dist"] = pd.to_numeric(puntos_df["dist"], errors="coerce")
    puntos_df.dropna(subset=["dist"], inplace=True)
    if puntos_df.empty:
        return "⚠️ Todas las distancias son inválidas.", None, None, None
    puntos_df.sort_values("dist", inplace=True)
    orden_tcid = puntos_df["tcid"].tolist()
    dist_dict = dict(zip(puntos_df["tcid"], puntos_df["dist"]))

    # ── parsear ficheros MOM ─────────────────────────────────────────────────
    df = _parse_files(files if isinstance(files, list) else [files])

    # Solo trenes que pasen por TODOS los tcid seleccionados
    trenes_validos = []
    for tr in df["train"].unique():
        tcids_tr = set(df[df["train"] == tr]["tcid"])
        if all(t in tcids_tr for t in orden_tcid):
            trenes_validos.append(tr)
    if not trenes_validos:
        return "⚠️ Ningún tren pasa por todos los puntos seleccionados.", None, None, None

    # ── calcular velocidades ────────────────────────────────────────────────
    vel_rows = []
    for tr in trenes_validos:
        dft = (df[df["train"] == tr]
                 .sort_values("time")
                 .drop_duplicates("tcid"))
        # Mantener sólo los puntos solicitados
        dft = dft[dft["tcid"].isin(orden_tcid)]
        dft["dist"] = dft["tcid"].map(dist_dict)
        for i in range(1, len(dft)):
            tc_prev  = dft.iloc[i-1]["tcid"]
            tc_curr  = dft.iloc[i]["tcid"]
            dist_prev = dft.iloc[i-1]["dist"]
            dist_curr = dft.iloc[i]["dist"]

            # Mid‑point para alinear ambos sentidos
            dist_mid = (dist_prev + dist_curr) / 2.0

            dist_delta = dist_curr - dist_prev          # Δdist (m)
            if dir_mad:
                dist_delta = abs(dist_delta)
            time_delta = (dft.iloc[i]["time"] - dft.iloc[i-1]["time"]).total_seconds()
            if time_delta == 0:
                vel = np.nan
            else:
                vel = abs(dist_delta) / time_delta * 3.6   # km/h

            seg_label = f"{tc_prev}-{tc_curr}"

            vel_rows.append({
                "train": tr,
                "segment": seg_label,
                "dist": dist_mid,
                "vel": vel
            })

    vel_df = pd.DataFrame(vel_rows)
    if vel_df.empty:
        return "⚠️ No se pudieron calcular velocidades.", None, None, None

    # Asegurar que la columna vel sea numérica continua
    plot_df = vel_df.copy()
    plot_df["vel"] = pd.to_numeric(plot_df["vel"], errors="coerce")
    plot_df["train"] = plot_df["train"].astype(str)
    plot_df["dist"] = pd.to_numeric(plot_df["dist"], errors="coerce")
    plot_df["segment"] = plot_df["segment"].astype(str)
    plot_df["dist_km"] = plot_df["dist"] / 1000.0

    # Etiqueta para cada distancia (usamos el primer label encontrado)
    label_map = {row["dist"]: row["segment"] for row in vel_df.to_dict("records")}

    # Media por distancia intermedia
    avg_df = (vel_df.groupby("dist", as_index=False)
                      .agg(vel=("vel", "mean"), segment=("segment", "first"))
             .assign(train="Media"))

    # ── construir figura matplotlib ────────────────────────────────────────
    line_df = pd.concat([vel_df, avg_df], ignore_index=True)
    line_df["dist_km"] = line_df["dist"] / 1000.0
    line_df = line_df.sort_values(["train", "dist_km"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for train, grp in line_df.groupby("train"):
        color = "k" if train == "Media" else None
        ax.plot(grp["dist_km"], grp["vel"], marker="o", label=train, color=color)

    # Eje X en km con etiquetas de segmento
    unique_dists = sorted(label_map.keys())
    xticks_km = [d / 1000.0 for d in unique_dists]
    xtick_labels = [label_map[d] for d in unique_dists]
    ax.set_xticks(xticks_km)
    ax.set_xticklabels(xtick_labels, rotation=90)
    ax.set_xlabel("Segmento (km medio)")
    ax.set_ylabel("Velocidad (km/h)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(title="Tren")
    fig.tight_layout()

    # Tabla para inspeccionar los datos representados
    table_df = line_df.rename(columns={
        "train": "Tren",
        "segment": "Segmento",
        "dist_km": "Dist (km)",
        "vel": "Vel (km/h)"
    })[["Tren", "Segmento", "Dist (km)", "Vel (km/h)"]]

    # Guardar CSV temporal para descarga
    import uuid, os, tempfile
    tmp_csv = os.path.join(tempfile.gettempdir(), f"velocidades_{uuid.uuid4().hex}.csv")
    table_df.to_csv(tmp_csv, index=False, encoding="utf-8")

    return "", fig, table_df, tmp_csv


# CALLBACK para copiar CSV al textbox
def cb_csv_to_text(csv_file):
    """
    Cuando el usuario sube un CSV, copia su contenido al cuadro de texto
    para que se pueda revisar o editar antes de calcular velocidades.
    """
    if not csv_file:
        return gr.update(value="")
    try:
        # Compatibilidad Gradio (<5 dict, ≥5 FileData)
        file_path = None
        if isinstance(csv_file, dict):
            file_path = csv_file.get("path") or csv_file.get("name")
        elif hasattr(csv_file, "path"):
            file_path = csv_file.path
        elif hasattr(csv_file, "name"):
            file_path = csv_file.name
        if file_path and Path(file_path).exists():
            return gr.update(value=Path(file_path).read_text(encoding="utf-8", errors="replace"))
        # Fallback: leer del file‑like
        if hasattr(csv_file, "read"):
            csv_file.seek(0)
            return gr.update(value=csv_file.read().decode("utf-8", errors="replace"))
    except Exception as e:
        return gr.update(value=f"⚠️ Error leyendo CSV: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="CRC Explorer") as demo:
    gr.Markdown("# CRCXplorer – Análisis de ficheros envioMOM")
    # Ajuste visual para Plotly: modebar arriba‑izquierda y sin wrap
    gr.HTML("""
    <style>
    .modebar-container {left: 0 !important; right: auto !important;}
    .modebar {flex-wrap: nowrap !important;}
    </style>
    """)
    files = gr.Files(
        label="Ficheros xml o xml.gz",
        file_count="multiple",
        file_types=None   # sin filtro → permite .gz y nombres sin extensión
    )

    with gr.Tabs():
        with gr.TabItem("Trenes"):
            # Selección de tren de referencia (ninguno por defecto)
            tren_sel = gr.Dropdown(
                label="Tren de referencia (muestra circuitos)",
                choices=[],
                value=None,
                interactive=True,
            )
            files.change(cb_update_trenes, inputs=files, outputs=tren_sel)

            # Circuitos que recorre el tren seleccionado (multiselección; todos marcados por defecto)
            circuit_sel = gr.Dropdown(
                label="Puntos de control / Circuitos",
                choices=[],
                value=[],
                multiselect=True,
                interactive=True,
            )
            # Actualizar lista de circuitos cuando cambien ficheros o tren
            tren_sel.change(cb_update_circuitos, inputs=[files, tren_sel], outputs=circuit_sel)
            files.change(cb_update_circuitos, inputs=[files, tren_sel], outputs=circuit_sel)

            btn = gr.Button("Generar gráfico")
            out_msg = gr.Markdown()
            out_plot = gr.Plot()
            btn.click(cb_plot, inputs=[files, tren_sel, circuit_sel], outputs=[out_msg, out_plot])

        with gr.TabItem("Alarmas"):
            filtro_alarm = gr.Text(label="Filtro (AND/&  OR/|  NOT/!)")
            btn_alarm = gr.Button("Filtrar alarmas")
            out_alarm = gr.HTML()
            btn_alarm.click(
                cb_alarmas,
                inputs=[files, filtro_alarm],
                outputs=out_alarm,
            )
        

        with gr.TabItem("Desvíos"):
            filtro_switch = gr.Text(label="Filtro (AND/&  OR/|  NOT/!)")
            btn_switch = gr.Button("Filtrar desvíos")
            out_switch = gr.HTML()
            btn_switch.click(
                cb_switch,
                inputs=[files, filtro_switch],
                outputs=out_switch,
            )

        with gr.TabItem("Señales"):
            filtro_signal = gr.Text(label="Filtro (AND/&  OR/|  NOT/!)")
            btn_signal = gr.Button("Filtrar señales")
            out_signal = gr.HTML()
            btn_signal.click(
                cb_signal,
                inputs=[files, filtro_signal],
                outputs=out_signal,
            )

        with gr.TabItem("Velocidades"):
            with gr.Row():
                csv_file = gr.File(
                    label="Subir CSV de puntos",
                    file_types=[".csv"],
                    scale=1,
                )
                csv_text = gr.Textbox(
                    label="Puntos (csv: cv_id,dist_metros)",
                    lines=5,
                    placeholder="CV-001,0\\nCV-002,250\\n...",
                    scale=2,
                )
                # Al subir CSV → volcar contenido al textbox
                csv_file.change(cb_csv_to_text, inputs=csv_file, outputs=csv_text)
            dir_mad = gr.Checkbox(label="Dirección Madrid", value=False)
            btn_vel = gr.Button("Calcular velocidades")
            out_msg_vel = gr.Markdown()
            out_plot_vel = gr.Plot(label="Perfil de velocidad")
            out_df_vel   = gr.Dataframe(headers=["Tren","Segmento","Dist (km)","Vel (km/h)"], interactive=False)
            out_csv_vel  = gr.File(label="Descargar CSV")
            btn_vel.click(
                cb_velocidades,
                inputs=[files, csv_text, csv_file, dir_mad],
                outputs=[out_msg_vel, out_plot_vel, out_df_vel, out_csv_vel],
            )


    gr.Markdown("---\nSube uno o varios `envioMOM*.xml(.gz)` y explora. Las pestañas:\n"
                "• **Trenes** – recorrido de circuitos\n"
                "• **Alarmas** – filtra mensajes Alarm con AND/OR/NOT.\n"
                "• **Desvíos** – filtra mensajes Switch (posición / enclavado).\n"
                "• **Señales** – filtra mensajes Signal (estado de indicación).\n"
                "• **Velocidades** – calcular perfiles de velocidad entre puntos seleccionados.")
    
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        debug=True          # sigue mostrando trazas
    )
