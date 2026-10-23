"""Experimentos reproducibles para el informe. Ejecutar con: python experimentos.py

Genera tablas en resultados/ y figuras en figuras/.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).parent
sys.path.insert(0, str(RAIZ / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D

from algoritmos import ALGORITMOS, alternativas_destino, dijkstra
from grafo import BODEGA, DESTINOS, RUTAS_ACTUALES, cargar_matriz, construir_grafo, tiempo_ruta, validar_rutas
from simulacion import ESCENARIOS, LIMITE_PUNTUAL, comparar, plan_actual, plan_optimo
from visual import posiciones

RES, FIG = RAIZ / "resultados", RAIZ / "figuras"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

VEHICULOS, HORAS = 8, 8.0
SEMILLAS = list(range(1, 11))  # 10 repeticiones por escenario
COLORES = {"Rutas actuales": "#e8833a", "Rutas Dijkstra": "#2563eb"}
COL_DEST = dict(zip(DESTINOS, ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]))

G = construir_grafo(cargar_matriz())
POS = posiciones(G)


def dibujar_red(ax, g, rutas=None, titulo=""):
    dobles = [(a, b) for a, b in g.edges if g.has_edge(b, a) and a < b]
    unicos = [(a, b) for a, b in g.edges if not g.has_edge(b, a)]
    nx.draw_networkx_edges(g, POS, edgelist=dobles, ax=ax, edge_color="#9aa5b1", arrows=False, width=1.2)
    nx.draw_networkx_edges(g, POS, edgelist=unicos, ax=ax, edge_color="#e8833a", width=1.8,
                           arrowstyle="-|>", arrowsize=14, node_size=420)
    etiquetas = {}
    for a, b in dobles:
        wa, wb = g[a][b]["weight"], g[b][a]["weight"]
        etiquetas[(a, b)] = f"{wa:g}" if wa == wb else f"{wa:g}/{wb:g}"
    for a, b in unicos:
        etiquetas[(a, b)] = f"{g[a][b]['weight']:g}"
    nx.draw_networkx_edge_labels(g, POS, edge_labels=etiquetas, ax=ax, font_size=6.5,
                                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    for i, ruta in enumerate((rutas or {}).values()):
        color = list(COL_DEST.values())[i % 4]
        nx.draw_networkx_edges(g, POS, edgelist=list(zip(ruta, ruta[1:])), ax=ax, edge_color=color, width=4,
                               alpha=0.6, arrowstyle="-|>", arrowsize=16, node_size=420)
    colores = ["#111827" if n == BODEGA else "#dc2626" if n in DESTINOS else "#e2e8f0" for n in g.nodes]
    nx.draw_networkx_nodes(g, POS, ax=ax, node_color=colores, node_size=420, edgecolors="#475569")
    nx.draw_networkx_labels(g, POS, ax=ax, font_size=7,
                            font_color={n: "white" if n == BODEGA or n in DESTINOS else "black" for n in g.nodes})
    ax.set_title(titulo)
    ax.axis("off")


# ------------------------------------------------------------------ 1. grafo
fig, ax = plt.subplots(figsize=(16, 13))
dibujar_red(ax, G, titulo="Red vial de Logística del Valle (tiempos en minutos)")
ax.legend(handles=[Line2D([], [], color="#9aa5b1", lw=2, label="doble sentido; «x/y» = Nmenor→Nmayor / Nmayor→Nmenor"),
                   Line2D([], [], color="#e8833a", lw=2, marker=">", label="un solo sentido"),
                   Line2D([], [], ls="", marker="o", mfc="#111827", mec="#475569", ms=10, label="bodega N01"),
                   Line2D([], [], ls="", marker="o", mfc="#dc2626", mec="#475569", ms=10, label="destinos")],
          loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "grafo.png", dpi=150)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(22, 10))
dibujar_red(axs[0], G, {d: RUTAS_ACTUALES[d] for d in DESTINOS}, "Rutas actuales (ida)")
dibujar_red(axs[1], G, {d: dijkstra(G, BODEGA, d)[0] for d in DESTINOS}, "Rutas Dijkstra (ida)")
axs[1].legend(handles=[Line2D([], [], color=c, lw=4, label=d) for d, c in COL_DEST.items()], loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "rutas_actual_vs_dijkstra.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ 2. validación y algoritmos
val = validar_rutas(G)
val.to_csv(RES / "validacion_rutas.csv", index=False)

filas = []
for d in DESTINOS:
    for sentido, (o, t) in {"Ida": (BODEGA, d), "Regreso": (d, BODEGA)}.items():
        for nombre, f in ALGORITMOS.items():
            ruta, costo, trabajo = f(G, o, t)
            filas.append({"Destino": d, "Sentido": sentido, "Algoritmo": nombre, "Tiempo": costo,
                          "Trabajo": trabajo, "Ruta": " → ".join(ruta)})
alg = pd.DataFrame(filas)
alg.to_csv(RES / "comparacion_algoritmos.csv", index=False)

# ------------------------------------------------------------------ 3. tabla de rutas por destino
filas = []
for d in DESTINOS:
    for nombre, ida, reg in alternativas_destino(G, d):
        ti, tr = tiempo_ruta(G, ida), tiempo_ruta(G, reg)
        filas.append({"Destino": d, "Alternativa": nombre, "Ida": " → ".join(ida), "Tiempo ida": ti,
                      "Regreso": " → ".join(reg), "Tiempo regreso": tr, "Ciclo": ti + tr,
                      "Ida invertida válida": tiempo_ruta(G, ida[::-1]) is not None,
                      "Puntual sin espera": ti <= LIMITE_PUNTUAL})
rutas = pd.DataFrame(filas)
rutas.to_csv(RES / "rutas_por_destino.csv", index=False)

pa, po = plan_actual(G).tabla(), plan_optimo(G).tabla()
fig, ax = plt.subplots(figsize=(9, 4.5))
x = range(len(DESTINOS))
for k, (tabla, nombre, desp) in enumerate([(pa, "Rutas actuales", -0.2), (po, "Rutas Dijkstra", 0.2)]):
    ax.bar([i + desp for i in x], tabla["Tiempo ida"], 0.38, color=COLORES[nombre], label=f"{nombre} — ida")
    ax.bar([i + desp for i in x], tabla["Tiempo regreso"], 0.38, bottom=tabla["Tiempo ida"],
           color=COLORES[nombre], alpha=0.45, label=f"{nombre} — regreso")
    for i, (ti, tc) in enumerate(zip(tabla["Tiempo ida"], tabla["Ciclo"])):
        ax.text(i + desp, tc + 2, f"{ti:g}+{tc - ti:g}", ha="center", fontsize=8)
ax.axhline(LIMITE_PUNTUAL, ls="--", color="gray")
ax.text(3.45, LIMITE_PUNTUAL + 2, "75 min", color="gray", fontsize=8)
ax.set_xticks(list(x), DESTINOS)
ax.set_ylabel("minutos")
ax.set_title("Tiempo de ida y regreso por destino")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ciclo_por_destino.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 4. experimentos por escenario
exp = pd.concat([comparar(G, t, VEHICULOS, HORAS, SEMILLAS, escenario=e) for e, t in ESCENARIOS.items()])
exp.to_csv(RES / "experimentos.csv", index=False)
METRICAS = ["Pedidos", "Entregas puntuales", "% puntual", "Espera promedio", "Solicitud→entrega promedio",
            "Utilización", "Tiempo adicional"]
resumen = exp.groupby(["Escenario", "Política"], sort=False)[METRICAS].agg(["mean", "std"]).round(1)
resumen.to_csv(RES / "resumen_escenarios.csv")

par = exp.pivot_table(index=["Escenario", "Semilla"], columns="Política", values=METRICAS[1:])
dif = pd.DataFrame({m: par[(m, "Rutas Dijkstra")] - par[(m, "Rutas actuales")] for m in METRICAS[1:]})
dif_res = dif.groupby("Escenario").agg(["mean", "min", "max"]).reindex(list(ESCENARIOS)).round(1)
dif_res.to_csv(RES / "diferencias_pareadas.csv")

fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
for ax, m in zip(axs.flat, METRICAS[1:]):
    datos, pos, cols = [], [], []
    for i, e in enumerate(ESCENARIOS):
        for j, p in enumerate(COLORES):
            datos.append(exp[(exp["Escenario"] == e) & (exp["Política"] == p)][m])
            pos.append(i * 3 + j)
            cols.append(COLORES[p])
    bp = ax.boxplot(datos, positions=pos, widths=0.8, patch_artist=True)
    for caja, c in zip(bp["boxes"], cols):
        caja.set_facecolor(c)
        caja.set_alpha(0.7)
    ax.set_xticks([i * 3 + 0.5 for i in range(3)], list(ESCENARIOS))
    ax.set_title(m)
    ax.grid(axis="y", alpha=0.3)
axs.flat[0].legend(handles=[Line2D([], [], color=c, lw=8, alpha=0.7, label=p) for p, c in COLORES.items()],
                   fontsize=8)
fig.suptitle(f"{len(SEMILLAS)} repeticiones pareadas por escenario · {VEHICULOS} vehículos · {HORAS:g} h")
fig.tight_layout()
fig.savefig(FIG / "experimentos_escenarios.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 5. barrido de flota
barrido = pd.concat([comparar(G, t, v, HORAS, SEMILLAS, escenario=e).assign(Vehículos=v)
                     for e, t in ESCENARIOS.items() for v in range(2, 25)])
barrido_m = barrido.groupby(["Escenario", "Política", "Vehículos"], as_index=False, sort=False)[METRICAS].mean()
barrido_m.to_csv(RES / "barrido_flota.csv", index=False)

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, m in zip(axs, ["% puntual", "Espera promedio", "Utilización"]):
    for e, ls in zip(ESCENARIOS, ["-", "--", ":"]):
        for p, c in COLORES.items():
            s = barrido_m[(barrido_m["Escenario"] == e) & (barrido_m["Política"] == p)]
            ax.plot(s["Vehículos"], s[m], ls=ls, color=c, marker=".", label=f"{e} · {p}")
    ax.axvline(VEHICULOS, color="gray", lw=0.8)
    ax.set_xlabel("vehículos")
    ax.set_title(m)
    ax.grid(alpha=0.3)
axs[1].set_yscale("symlog", linthresh=10)
axs[1].set_ylim(bottom=0)
axs[0].legend(fontsize=7)
fig.suptitle("Efecto del tamaño de la flota (promedio de 10 semillas; línea gris = flota actual)")
fig.tight_layout()
fig.savefig(FIG / "barrido_flota.png", dpi=150)
plt.close(fig)

# flota mínima para ≥ 90 % de entregas puntuales
flota_min = (barrido_m[barrido_m["% puntual"] >= 90].groupby(["Escenario", "Política"], sort=False)["Vehículos"]
             .min().unstack().reindex(index=list(ESCENARIOS), columns=list(COLORES)))
flota_min = flota_min.apply(lambda c: c.map(lambda v: "nunca (> 24)" if pd.isna(v) else int(v)))
flota_min.to_csv(RES / "flota_minima_90.csv")

# ------------------------------------------------------------------ 6. incidentes viales
INCIDENTES = {
    "Red original": {},
    "Congestión N01→N17 (8→25 min)": {("N01", "N17"): 25},
    "Cierre N29↔N31": {("N29", "N31"): None, ("N31", "N29"): None},
    "Cierre N05↔N06": {("N05", "N06"): None, ("N06", "N05"): None},
}
filas_r, filas_s = [], []
for nombre, cambios in INCIDENTES.items():
    g = G.copy()
    for (a, b), t in cambios.items():
        if t is None:
            g.remove_edge(a, b)
        else:
            g[a][b]["weight"] = t
    for pol, plan in {"Rutas actuales": plan_actual(g), "Rutas Dijkstra": plan_optimo(g)}.items():
        for d in DESTINOS:
            filas_r.append({"Incidente": nombre, "Política": pol, "Destino": d,
                            "Ida": " → ".join(plan.ida[d]), "Tiempo ida": plan.t_ida[d],
                            "Regreso": " → ".join(plan.regreso[d]), "Tiempo regreso": plan.t_regreso[d]})
    r = comparar(g, ESCENARIOS["Normal"], VEHICULOS, HORAS, SEMILLAS, escenario=nombre)
    filas_s.append(r.groupby("Política", as_index=False)[METRICAS].mean().assign(Incidente=nombre))
pd.DataFrame(filas_r).to_csv(RES / "incidentes_rutas.csv", index=False)
inc = pd.concat(filas_s)[["Incidente", "Política", *METRICAS]].round(1)
inc.to_csv(RES / "incidentes_simulacion.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 80)
print(val, alg, rutas, resumen, dif_res, flota_min, inc, pd.DataFrame(filas_r), sep="\n\n")
