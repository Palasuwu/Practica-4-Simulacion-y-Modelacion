"""Algoritmos de búsqueda de rutas implementados a mano: Dijkstra, A* y Bellman-Ford.

Todos devuelven (ruta, costo, nodos_expandidos) para poder compararlos.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque

import networkx as nx

from grafo import RUTAS_ACTUALES

INF = float("inf")


def _reconstruir(previo: dict, origen: str, destino: str) -> list[str]:
    if destino != origen and destino not in previo:
        return []
    ruta = [destino]
    while ruta[-1] != origen:
        ruta.append(previo[ruta[-1]])
    return ruta[::-1]


def dijkstra(g: nx.DiGraph, origen: str, destino: str) -> tuple[list[str], float, int]:
    dist = {origen: 0.0}
    previo: dict[str, str] = {}
    visitados = set()
    cola = [(0.0, origen)]
    while cola:
        d, u = heapq.heappop(cola)
        if u in visitados:
            continue
        visitados.add(u)
        if u == destino:
            break
        for v, datos in g[u].items():
            nd = d + datos["weight"]
            if nd < dist.get(v, INF):
                dist[v] = nd
                previo[v] = u
                heapq.heappush(cola, (nd, v))
    if destino not in dist:
        return [], INF, len(visitados)
    return _reconstruir(previo, origen, destino), dist[destino], len(visitados)


def heuristica_saltos(g: nx.DiGraph, destino: str) -> dict[str, float]:
    """h(n) = (mínimo número de tramos de n al destino) × (tiempo del tramo más corto de la red).

    La red no tiene coordenadas, así que no hay distancia euclidiana. Esta cota nunca
    sobreestima el costo real (cada tramo cuesta al menos el mínimo), por lo que es
    admisible y consistente: A* devuelve la misma ruta óptima que Dijkstra, solo
    explorando menos nodos.
    """
    peso_min = min(d["weight"] for _, _, d in g.edges(data=True))
    saltos = {destino: 0}
    cola = deque([destino])
    while cola:  # BFS sobre el grafo invertido
        v = cola.popleft()
        for u in g.predecessors(v):
            if u not in saltos:
                saltos[u] = saltos[v] + 1
                cola.append(u)
    return {n: saltos.get(n, INF) * peso_min for n in g.nodes}


def a_estrella(g: nx.DiGraph, origen: str, destino: str) -> tuple[list[str], float, int]:
    h = heuristica_saltos(g, destino)
    if h[origen] == INF:
        return [], INF, 0
    gcost = {origen: 0.0}
    previo: dict[str, str] = {}
    cerrados = set()
    contador = itertools.count()
    cola = [(h[origen], next(contador), origen)]
    while cola:
        _, _, u = heapq.heappop(cola)
        if u in cerrados:
            continue
        cerrados.add(u)
        if u == destino:
            break
        for v, datos in g[u].items():
            ng = gcost[u] + datos["weight"]
            if ng < gcost.get(v, INF):
                gcost[v] = ng
                previo[v] = u
                heapq.heappush(cola, (ng + h[v], next(contador), v))
    if destino not in gcost:
        return [], INF, len(cerrados)
    return _reconstruir(previo, origen, destino), gcost[destino], len(cerrados)


def bellman_ford(g: nx.DiGraph, origen: str, destino: str) -> tuple[list[str], float, int]:
    """Relaja todos los arcos |V|-1 veces; devuelve las relajaciones como medida de trabajo."""
    dist = {n: INF for n in g.nodes}
    dist[origen] = 0.0
    previo: dict[str, str] = {}
    relajaciones = 0
    for _ in range(g.number_of_nodes() - 1):
        cambio = False
        for u, v, datos in g.edges(data=True):
            relajaciones += 1
            if dist[u] + datos["weight"] < dist[v]:
                dist[v] = dist[u] + datos["weight"]
                previo[v] = u
                cambio = True
        if not cambio:
            break
    if dist[destino] == INF:
        return [], INF, relajaciones
    return _reconstruir(previo, origen, destino), dist[destino], relajaciones


ALGORITMOS = {"Dijkstra": dijkstra, "A*": a_estrella, "Bellman-Ford": bellman_ford}


def rutas_alternativas(g: nx.DiGraph, origen: str, destino: str, k: int = 3) -> list[list[str]]:
    """Las k rutas simples más cortas (algoritmo de Yen, vía networkx)."""
    try:
        return list(itertools.islice(nx.shortest_simple_paths(g, origen, destino, weight="weight"), k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def ruta_menos_tramos(g: nx.DiGraph, origen: str, destino: str) -> list[str]:
    """Ruta con menos intersecciones (BFS, ignora tiempos): alternativa 'intuitiva'."""
    try:
        return nx.shortest_path(g, origen, destino)
    except nx.NetworkXNoPath:
        return []


def reparar_ruta(g: nx.DiGraph, ruta: list[str]) -> list[str]:
    """Si un tramo de la ruta fija dejó de existir, el conductor lo rodea por el desvío
    más corto entre los extremos del tramo y luego retoma su ruta habitual."""
    if len(ruta) < 2:
        return ruta
    nueva = [ruta[0]]
    for a, b in zip(ruta, ruta[1:]):
        if g.has_edge(a, b):
            nueva.append(b)
        else:
            desvio, costo, _ = dijkstra(g, a, b)
            if not desvio:
                return []
            nueva.extend(desvio[1:])
    return nueva


def alternativas_destino(g: nx.DiGraph, destino: str, origen: str = "N01",
                         ruta_actual: list[str] | None = None) -> list[tuple[str, list[str], list[str]]]:
    """Candidatas para comparar: la ruta actual (regreso invertido), la óptima de Dijkstra y la
    2.ª y 3.ª más cortas de Yen, todas con ida y regreso calculados por separado para que el
    regreso sea siempre válido aunque haya calles de un solo sentido."""
    actual = ruta_actual or RUTAS_ACTUALES[destino]
    idas = [p for p in rutas_alternativas(g, origen, destino, 4) if p != actual]
    regresos = [p for p in rutas_alternativas(g, destino, origen, 4) if p != actual[::-1]]
    ida_d, reg_d = dijkstra(g, origen, destino)[0], dijkstra(g, destino, origen)[0]
    idas = [p for p in idas if p != ida_d]
    regresos = [p for p in regresos if p != reg_d]
    candidatas = [("Actual (hoja de despacho)", actual, actual[::-1]),
                  ("Dijkstra (óptima)", ida_d, reg_d)]
    for k, nombre in enumerate(["2.ª más corta (Yen)", "3.ª más corta (Yen)"]):
        if k < len(idas) and k < len(regresos):
            candidatas.append((nombre, idas[k], regresos[k]))
    return candidatas
