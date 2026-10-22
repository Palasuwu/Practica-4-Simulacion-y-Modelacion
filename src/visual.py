"""Gráficas con Plotly para la app."""
from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from grafo import BODEGA, DESTINOS

COLOR_DOBLE = "#9aa5b1"
COLOR_UNICO = "#e8833a"
COLOR_RUTAS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]


def posiciones(g_base: nx.DiGraph) -> dict:
    """Disposición fija calculada sobre la red original para que no cambie con incidentes."""
    return nx.kamada_kawai_layout(g_base.to_undirected(), weight=None)


def figura_grafo(g: nx.DiGraph, pos: dict, rutas: dict[str, list[str]] | None = None,
                 cerrados: list[tuple[str, str]] | None = None, mostrar_pesos: bool = True,
                 altura: int = 700) -> go.Figure:
    """Arcos de doble sentido en gris (línea recta con el peso de cada sentido), arcos de un
    solo sentido en naranja con flecha, rutas resaltadas encima y tramos cerrados en rojo punteado."""
    fig = go.Figure()
    anotaciones = []
    dibujados = set()
    for a, b, d in g.edges(data=True):
        if (b, a) in dibujados:
            continue
        dibujados.add((a, b))
        (x0, y0), (x1, y1) = pos[a], pos[b]
        doble = g.has_edge(b, a)
        if doble:
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip",
                                     line=dict(color=COLOR_DOBLE, width=1.5), showlegend=False))
            if mostrar_pesos:
                wa, wb = d["weight"], g[b][a]["weight"]
                txt = f"{wa:g}" if wa == wb else f"{a[1:]}→{b[1:]}: {wa:g} / {b[1:]}→{a[1:]}: {wb:g}"
                anotaciones.append(dict(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=txt, showarrow=False,
                                        font=dict(size=9, color="#334155" if wa == wb else "#b45309"),
                                        bgcolor="rgba(255,255,255,0.7)"))
        else:
            anotaciones.append(dict(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                                    showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
                                    arrowcolor=COLOR_UNICO, standoff=11))
            if mostrar_pesos:
                anotaciones.append(dict(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"{d['weight']:g}",
                                        showarrow=False, font=dict(size=9, color=COLOR_UNICO),
                                        bgcolor="rgba(255,255,255,0.7)"))

    for a, b in cerrados or []:
        if a in pos and b in pos:
            (x0, y0), (x1, y1) = pos[a], pos[b]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip",
                                     line=dict(color="red", width=3, dash="dot"), showlegend=False))

    for i, (nombre, ruta) in enumerate((rutas or {}).items()):
        if len(ruta) < 2:
            continue
        color = COLOR_RUTAS[i % len(COLOR_RUTAS)]
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in ruta], y=[pos[n][1] for n in ruta],
                                 mode="lines", line=dict(color=color, width=5), opacity=0.55,
                                 name=nombre, hoverinfo="name"))
        for a, b in zip(ruta, ruta[1:]):
            anotaciones.append(dict(x=pos[b][0], y=pos[b][1], ax=pos[a][0], ay=pos[a][1],
                                    xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                    arrowhead=2, arrowsize=1, arrowwidth=2.5, arrowcolor=color,
                                    standoff=11, opacity=0.9))

    colores = ["#111827" if n == BODEGA else "#dc2626" if n in DESTINOS else "#e2e8f0" for n in g.nodes]
    texto = ["white" if n == BODEGA or n in DESTINOS else "#0f172a" for n in g.nodes]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in g.nodes], y=[pos[n][1] for n in g.nodes], mode="markers+text",
        text=list(g.nodes), textfont=dict(size=10, color=texto),
        marker=dict(size=26, color=colores, line=dict(color="#475569", width=1)),
        hovertext=[f"{n}<br>sale a: {', '.join(g.successors(n))}<br>llega de: {', '.join(g.predecessors(n))}"
                   for n in g.nodes],
        hoverinfo="text", showlegend=False))
    fig.update_layout(annotations=anotaciones, height=altura, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      plot_bgcolor="white", legend=dict(orientation="h", y=1.03))
    return fig


def figura_cola(df: pd.DataFrame, horas: float, titulo: str) -> go.Figure:
    """Pedidos en cola y vehículos en ruta a lo largo del tiempo (funciones escalonadas)."""
    def escalon(eventos):
        ev = sorted(eventos)
        t, y, nivel = [0.0], [0], 0
        for tt, delta in ev:
            nivel += delta
            t.append(tt)
            y.append(nivel)
        return t, y

    tc, yc = escalon([(r.Llegada, 1) for r in df.itertuples()] + [(r.Salida, -1) for r in df.itertuples()])
    tv, yv = escalon([(r.Salida, 1) for r in df.itertuples()] + [(r.Regreso, -1) for r in df.itertuples()])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tc, y=yc, line_shape="hv", name="Pedidos en cola"))
    fig.add_trace(go.Scatter(x=tv, y=yv, line_shape="hv", name="Vehículos en ruta"))
    fig.add_vline(x=horas * 60, line_dash="dash", line_color="gray", annotation_text="cierre de recepción")
    fig.update_layout(title=titulo, xaxis_title="minuto de la jornada", yaxis_title="cantidad",
                      height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return fig


def figura_gantt(df: pd.DataFrame, horas: float, titulo: str) -> go.Figure:
    """Ocupación de cada vehículo: ida (color del destino) y regreso (gris)."""
    fig = go.Figure()
    colores = dict(zip(DESTINOS, COLOR_RUTAS))
    vistos = set()
    for r in df.itertuples():
        fig.add_trace(go.Bar(base=[r.Salida], x=[r.Entrega - r.Salida], y=[f"V{r.Vehículo}"], orientation="h",
                             marker_color=colores[r.Destino], name=r.Destino,
                             showlegend=r.Destino not in vistos,
                             hovertext=f"Pedido {r.Pedido} → {r.Destino}<br>espera {r.Espera:.1f} min",
                             hoverinfo="text"))
        vistos.add(r.Destino)
        fig.add_trace(go.Bar(base=[r.Entrega], x=[r.Regreso - r.Entrega], y=[f"V{r.Vehículo}"], orientation="h",
                             marker_color="#cbd5e1", showlegend=False, hoverinfo="skip"))
    fig.add_vline(x=horas * 60, line_dash="dash", line_color="gray")
    fig.update_layout(title=titulo, barmode="overlay", height=60 + 28 * df["Vehículo"].nunique(),
                      xaxis_title="minuto de la jornada", margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(categoryorder="category descending"), legend=dict(orientation="h", y=-0.25))
    return fig
