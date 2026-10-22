"""Construcción del grafo dirigido a partir de la matriz de adyacencia y validación de rutas."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

RUTA_MATRIZ = Path(__file__).resolve().parent.parent / "data" / "matriz.csv"

BODEGA = "N01"
DESTINOS = ["N08", "N19", "N34", "N40"]

# Rutas de ida registradas en las hojas de despacho.
RUTAS_ACTUALES = {
    "N08": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08"],
    "N19": ["N01", "N02", "N03", "N11", "N19"],
    "N34": ["N01", "N09", "N17", "N25", "N33", "N34"],
    "N40": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N16", "N24", "N32", "N40"],
}


def cargar_matriz(ruta: Path | str = RUTA_MATRIZ) -> pd.DataFrame:
    """Lee la matriz; las celdas vacías (sin conexión) quedan como NaN."""
    return pd.read_csv(ruta, index_col=0)


def construir_grafo(matriz: pd.DataFrame) -> nx.DiGraph:
    """Cada celda (i, j) con valor > 0 es un arco dirigido i -> j con peso en minutos.

    Una calle de doble sentido aparece como dos arcos (i->j y j->i), posiblemente con
    tiempos distintos; una calle de un solo sentido aparece como un único arco.
    """
    g = nx.DiGraph()
    g.add_nodes_from(matriz.index)
    for origen in matriz.index:
        for destino, valor in matriz.loc[origen].items():
            if pd.notna(valor) and origen != destino and float(valor) > 0:
                g.add_edge(origen, destino, weight=float(valor))
    return g


def tiempo_ruta(g: nx.DiGraph, ruta: list[str]) -> float | None:
    """Suma de tiempos de la ruta; None si algún tramo no existe."""
    total = 0.0
    for a, b in zip(ruta, ruta[1:]):
        if not g.has_edge(a, b):
            return None
        total += g[a][b]["weight"]
    return total


def tramos_faltantes(g: nx.DiGraph, ruta: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for a, b in zip(ruta, ruta[1:]) if not g.has_edge(a, b)]


def validar_rutas(g: nx.DiGraph, rutas: dict[str, list[str]] = RUTAS_ACTUALES) -> pd.DataFrame:
    """Valida cada ruta de ida y su regreso por el mismo camino en sentido inverso."""
    filas = []
    for destino, ida in rutas.items():
        regreso = list(reversed(ida))
        filas.append({
            "Destino": destino,
            "Ruta ida": " → ".join(ida),
            "Ida válida": not tramos_faltantes(g, ida),
            "Tiempo ida (min)": tiempo_ruta(g, ida),
            "Tramos faltantes ida": ", ".join(f"{a}→{b}" for a, b in tramos_faltantes(g, ida)) or "—",
            "Regreso válido": not tramos_faltantes(g, regreso),
            "Tiempo regreso (min)": tiempo_ruta(g, regreso),
            "Tramos faltantes regreso": ", ".join(f"{a}→{b}" for a, b in tramos_faltantes(g, regreso)) or "—",
        })
    return pd.DataFrame(filas)


def resumen_sentidos(g: nx.DiGraph) -> dict:
    """Cuenta pares con doble sentido (simétricos o no) y arcos de un solo sentido."""
    doble_sim, doble_asim, unico = 0, 0, []
    vistos = set()
    for a, b, d in g.edges(data=True):
        if (b, a) in vistos:
            continue
        vistos.add((a, b))
        if g.has_edge(b, a):
            if g[b][a]["weight"] == d["weight"]:
                doble_sim += 1
            else:
                doble_asim += 1
        else:
            unico.append((a, b))
    return {
        "arcos": g.number_of_edges(),
        "pares_doble_sentido_simetricos": doble_sim,
        "pares_doble_sentido_asimetricos": doble_asim,
        "arcos_un_sentido": unico,
    }
