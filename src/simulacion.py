"""Simulación de eventos discretos de la operación de despacho.

Eventos: LLEGADA de pedido, ENTREGA en destino y REGRESO del vehículo a la bodega.
Los pedidos se generan una sola vez por semilla (proceso de Poisson) y se reutilizan
para ambas políticas de rutas (números aleatorios comunes).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from algoritmos import dijkstra, reparar_ruta
from grafo import BODEGA, DESTINOS, RUTAS_ACTUALES, tiempo_ruta

LIMITE_PUNTUAL = 75.0

ESCENARIOS = {
    "Normal": 4.0,
    "Semana Santa": 6.0,
    "Navidad": 10.0,
}


@dataclass
class Pedido:
    id: int
    llegada: float
    destino: str
    vehiculo: int | None = None
    salida: float | None = None
    entrega: float | None = None
    regreso: float | None = None

    @property
    def espera(self) -> float:
        return self.salida - self.llegada

    @property
    def tiempo_total(self) -> float:
        return self.entrega - self.llegada


def generar_pedidos(tasa_hora: float, horas: float, semilla: int,
                    destinos: list[str] = DESTINOS) -> list[Pedido]:
    """Tiempos entre llegadas ~ Exponencial(media = 60 / tasa) minutos; destino uniforme.

    Se usan dos flujos aleatorios independientes (uno para tiempos, otro para destinos)
    derivados de la misma semilla, para que cambiar uno no altere el otro.
    """
    flujo_t, flujo_d = (np.random.default_rng(s) for s in np.random.SeedSequence(semilla).spawn(2))
    fin = horas * 60.0
    pedidos, t = [], 0.0
    while True:
        t += flujo_t.exponential(60.0 / tasa_hora)
        if t > fin:
            break
        pedidos.append(Pedido(len(pedidos) + 1, t, destinos[flujo_d.integers(len(destinos))]))
    return pedidos


@dataclass
class PlanRutas:
    """Ruta de ida y de regreso por destino, con sus tiempos."""
    ida: dict[str, list[str]] = field(default_factory=dict)
    regreso: dict[str, list[str]] = field(default_factory=dict)
    t_ida: dict[str, float] = field(default_factory=dict)
    t_regreso: dict[str, float] = field(default_factory=dict)

    @property
    def factible(self) -> bool:
        return all(np.isfinite(self.t_ida[d]) and np.isfinite(self.t_regreso[d]) for d in self.ida)

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Destino": d,
            "Ida": " → ".join(self.ida[d]) or "SIN RUTA",
            "Tiempo ida": self.t_ida[d],
            "Regreso": " → ".join(self.regreso[d]) or "SIN RUTA",
            "Tiempo regreso": self.t_regreso[d],
            "Ciclo": self.t_ida[d] + self.t_regreso[d],
        } for d in self.ida])


def plan_actual(g: nx.DiGraph) -> PlanRutas:
    """Política 1: rutas de las hojas de despacho; regreso por el mismo camino invertido.
    Si un tramo está cerrado, el conductor toma el desvío más corto para ese tramo."""
    plan = PlanRutas()
    for d, ruta in RUTAS_ACTUALES.items():
        ida = reparar_ruta(g, ruta)
        reg = reparar_ruta(g, list(reversed(ruta)))
        plan.ida[d], plan.regreso[d] = ida, reg
        plan.t_ida[d] = tiempo_ruta(g, ida) if ida else float("inf")
        plan.t_regreso[d] = tiempo_ruta(g, reg) if reg else float("inf")
    return plan


def plan_optimo(g: nx.DiGraph, algoritmo=dijkstra) -> PlanRutas:
    """Política 2: ruta más corta de ida y de regreso, calculadas por separado
    (la red es asimétrica, así que el mejor regreso no es la ida invertida)."""
    plan = PlanRutas()
    for d in DESTINOS:
        plan.ida[d], plan.t_ida[d], _ = algoritmo(g, BODEGA, d)
        plan.regreso[d], plan.t_regreso[d], _ = algoritmo(g, d, BODEGA)
    return plan


def simular(pedidos: list[Pedido], plan: PlanRutas, vehiculos: int, horas: float,
            t_descarga: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Ejecuta la simulación y devuelve el detalle por pedido y las métricas."""
    pedidos = [Pedido(p.id, p.llegada, p.destino) for p in pedidos]  # copia limpia
    eventos: list[tuple[float, int, str, int]] = []  # (tiempo, prioridad, tipo, índice)
    for i, p in enumerate(pedidos):
        heapq.heappush(eventos, (p.llegada, 1, "LLEGADA", i))

    libres = list(range(vehiculos))  # vehículos en bodega (el de menor número sale primero)
    cola: list[int] = []  # FIFO
    ocupado = [0.0] * vehiculos

    def despachar(t: float):
        while cola and libres:
            i = cola.pop(0)
            v = heapq.heappop(libres)
            p = pedidos[i]
            p.vehiculo, p.salida = v + 1, t
            p.entrega = t + plan.t_ida[p.destino] + t_descarga
            p.regreso = p.entrega + plan.t_regreso[p.destino]
            ocupado[v] += p.regreso - t
            heapq.heappush(eventos, (p.entrega, 2, "ENTREGA", i))
            # Prioridad 0: un vehículo que regresa al mismo instante que llega un pedido
            # queda disponible para atenderlo.
            heapq.heappush(eventos, (p.regreso, 0, "REGRESO", i))

    while eventos:
        t, _, tipo, i = heapq.heappop(eventos)
        if tipo == "LLEGADA":
            cola.append(i)
        elif tipo == "REGRESO":
            heapq.heappush(libres, pedidos[i].vehiculo - 1)
        despachar(t)

    df = pd.DataFrame([{
        "Pedido": p.id, "Destino": p.destino, "Llegada": p.llegada, "Vehículo": p.vehiculo,
        "Salida": p.salida, "Espera": p.espera, "Entrega": p.entrega,
        "Solicitud→entrega": p.tiempo_total, "Regreso": p.regreso,
        "Puntual": p.tiempo_total <= LIMITE_PUNTUAL,
    } for p in pedidos])
    return df, metricas(df, vehiculos, horas, ocupado)


def metricas(df: pd.DataFrame, vehiculos: int, horas: float, ocupado: list[float]) -> dict:
    fin_recepcion = horas * 60.0
    if df.empty:
        return {"Pedidos": 0, "Entregas puntuales": 0, "% puntual": 0.0, "Espera promedio": 0.0,
                "Espera máxima": 0.0, "Solicitud→entrega promedio": 0.0, "Utilización": 0.0,
                "Fin de operación": fin_recepcion, "Tiempo adicional": 0.0}
    fin = max(fin_recepcion, df["Regreso"].max())
    return {
        "Pedidos": len(df),
        "Entregas puntuales": int(df["Puntual"].sum()),
        "% puntual": 100.0 * df["Puntual"].mean(),
        "Espera promedio": df["Espera"].mean(),
        "Espera máxima": df["Espera"].max(),
        "Solicitud→entrega promedio": df["Solicitud→entrega"].mean(),
        "Utilización": 100.0 * sum(ocupado) / (vehiculos * fin),
        "Fin de operación": fin,
        # Minutos después del cierre de recepción hasta que el último vehículo vuelve a N01.
        "Tiempo adicional": fin - fin_recepcion,
    }


def comparar(g: nx.DiGraph, tasa: float, vehiculos: int, horas: float, semillas: list[int],
             t_descarga: float = 0.0, escenario: str = "") -> pd.DataFrame:
    """Corre ambas políticas con los mismos pedidos para cada semilla."""
    planes = {"Rutas actuales": plan_actual(g), "Rutas Dijkstra": plan_optimo(g)}
    filas = []
    for s in semillas:
        pedidos = generar_pedidos(tasa, horas, s)
        for nombre, plan in planes.items():
            if not plan.factible:
                continue
            _, m = simular(pedidos, plan, vehiculos, horas, t_descarga)
            filas.append({"Escenario": escenario, "Semilla": s, "Política": nombre, **m})
    return pd.DataFrame(filas)
