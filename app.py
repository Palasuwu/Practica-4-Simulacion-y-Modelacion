"""Logística del Valle — estudio de rutas y flota. Ejecutar con: streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from algoritmos import ALGORITMOS, dijkstra, rutas_alternativas, ruta_menos_tramos
from grafo import (BODEGA, DESTINOS, RUTAS_ACTUALES, cargar_matriz, construir_grafo,
                   resumen_sentidos, tiempo_ruta, validar_rutas)
from simulacion import (ESCENARIOS, LIMITE_PUNTUAL, comparar, generar_pedidos, plan_actual,
                        plan_optimo, simular)
from visual import figura_cola, figura_gantt, figura_grafo, posiciones

st.set_page_config(page_title="Logística del Valle", page_icon="🚚", layout="wide")


@st.cache_data
def datos_base():
    matriz = cargar_matriz()
    g = construir_grafo(matriz)
    return matriz, g, posiciones(g)


matriz, g_base, POS = datos_base()

if "incidentes" not in st.session_state:
    st.session_state.incidentes = {}  # (origen, destino) -> nuevo tiempo, o None si está cerrado


def red_actual():
    """Copia de la red original con los incidentes aplicados."""
    g = g_base.copy()
    for (a, b), t in st.session_state.incidentes.items():
        if t is None:
            if g.has_edge(a, b):
                g.remove_edge(a, b)
        else:
            g.add_edge(a, b, weight=float(t))
    return g


# ---------------------------------------------------------------- barra lateral
st.sidebar.title("⚙️ Parámetros")
escenario = st.sidebar.radio("Escenario", [*ESCENARIOS, "Personalizado"], horizontal=False)
tasa = st.sidebar.number_input("Pedidos por hora (promedio)", 0.5, 40.0,
                               ESCENARIOS.get(escenario, 4.0), 0.5, disabled=escenario != "Personalizado")
vehiculos = st.sidebar.slider("Número de vehículos", 1, 30, 8)
horas = st.sidebar.slider("Duración de la recepción (horas)", 1.0, 16.0, 8.0, 0.5)
semilla = st.sidebar.number_input("Semilla aleatoria", 0, 10_000_000, 42, 1)
t_descarga = st.sidebar.number_input("Tiempo de entrega en destino (min)", 0.0, 30.0, 0.0, 1.0,
                                     help="Tiempo de descarga; 0 = solo se cuenta el viaje.")

st.sidebar.divider()
st.sidebar.subheader("🚧 Incidentes viales")
arcos = sorted(f"{a}→{b}" for a, b in g_base.edges)
sel = st.sidebar.selectbox("Tramo", arcos)
a_sel, b_sel = sel.split("→")
original = g_base[a_sel][b_sel]["weight"]
accion = st.sidebar.radio("Acción", ["Cambiar tiempo", "Cerrar tramo"], horizontal=True)
nuevo_t = st.sidebar.number_input(f"Nuevo tiempo (original {original:g} min)", 1.0, 300.0,
                                  float(original), 1.0, disabled=accion == "Cerrar tramo")
ambos = st.sidebar.checkbox("Aplicar también al sentido contrario", value=g_base.has_edge(b_sel, a_sel),
                            disabled=not g_base.has_edge(b_sel, a_sel))
c1, c2 = st.sidebar.columns(2)
if c1.button("Aplicar", width="stretch"):
    pares = [(a_sel, b_sel)] + ([(b_sel, a_sel)] if ambos and g_base.has_edge(b_sel, a_sel) else [])
    for p in pares:
        if accion == "Cerrar tramo":
            st.session_state.incidentes[p] = None
        elif accion == "Cambiar tiempo" and nuevo_t != g_base[p[0]][p[1]]["weight"]:
            st.session_state.incidentes[p] = nuevo_t
        else:
            st.session_state.incidentes.pop(p, None)
if c2.button("Restaurar red", width="stretch"):
    st.session_state.incidentes = {}
for (a, b), t in st.session_state.incidentes.items():
    st.sidebar.caption(f"• {a}→{b}: " + ("**CERRADO**" if t is None else f"{g_base[a][b]['weight']:g} → **{t:g} min**"))

G = red_actual()
cerrados = [k for k, v in st.session_state.incidentes.items() if v is None]
PLAN_ACT, PLAN_OPT = plan_actual(G), plan_optimo(G)

# ---------------------------------------------------------------- encabezado
st.title("🚚 Logística del Valle — rutas y flota")
st.caption(f"Bodega {BODEGA} · destinos {', '.join(DESTINOS)} · entrega puntual ≤ {LIMITE_PUNTUAL:g} min "
           f"desde la solicitud · escenario **{escenario}** ({tasa:g} pedidos/h)")
if st.session_state.incidentes:
    st.warning(f"Red modificada por {len(st.session_state.incidentes)} incidente(s). "
               "Las rutas Dijkstra se recalculan sobre la red modificada y las rutas actuales "
               "toman el desvío más corto en cada tramo cerrado.")

tab_red, tab_rutas, tab_sim, tab_exp = st.tabs(["🗺️ Red vial", "🧭 Rutas", "🎲 Simulación", "🧪 Experimentos"])

# ---------------------------------------------------------------- red
with tab_red:
    col_g, col_i = st.columns([3, 1])
    with col_i:
        resaltar = st.radio("Resaltar", ["Nada", "Rutas actuales (ida)", "Rutas Dijkstra (ida)",
                                         "Rutas Dijkstra (regreso)"])
        pesos = st.checkbox("Mostrar tiempos", True)
        r = resumen_sentidos(G)
        st.metric("Nodos", G.number_of_nodes())
        st.metric("Arcos dirigidos", r["arcos"])
        st.write(f"- Doble sentido, mismo tiempo: **{r['pares_doble_sentido_simetricos']}** pares\n"
                 f"- Doble sentido, tiempos distintos: **{r['pares_doble_sentido_asimetricos']}** pares\n"
                 f"- Un solo sentido: **{len(r['arcos_un_sentido'])}** arcos")
        st.caption("Gris = doble sentido (si el tiempo difiere se muestran ambos en naranja oscuro). "
                   "Naranja con flecha = un solo sentido. Negro = bodega, rojo = destinos.")
    rutas = {"Rutas actuales (ida)": {f"{d} actual": PLAN_ACT.ida[d] for d in DESTINOS},
             "Rutas Dijkstra (ida)": {f"{d} Dijkstra": PLAN_OPT.ida[d] for d in DESTINOS},
             "Rutas Dijkstra (regreso)": {f"{d} regreso": PLAN_OPT.regreso[d] for d in DESTINOS}}.get(resaltar)
    col_g.plotly_chart(figura_grafo(G, POS, rutas, cerrados, pesos), width="stretch")

    st.subheader("Validación de las rutas registradas (ida y regreso invertido)")
    st.dataframe(validar_rutas(G), hide_index=True, width="stretch")
    with st.expander("Matriz de adyacencia (minutos)"):
        st.dataframe(matriz, width="stretch")

# ---------------------------------------------------------------- rutas
with tab_rutas:
    st.subheader("Comparación de algoritmos")
    filas = []
    for d in DESTINOS:
        for sentido, (o, t) in {"Ida": (BODEGA, d), "Regreso": (d, BODEGA)}.items():
            for nombre, f in ALGORITMOS.items():
                ruta, costo, trabajo = f(G, o, t)
                filas.append({"Destino": d, "Sentido": sentido, "Algoritmo": nombre, "Tiempo (min)": costo,
                              "Nodos expandidos / relajaciones": trabajo, "Ruta": " → ".join(ruta) or "SIN RUTA"})
    df_alg = pd.DataFrame(filas)
    st.dataframe(df_alg, hide_index=True, width="stretch")
    st.plotly_chart(px.bar(df_alg, x="Destino", y="Nodos expandidos / relajaciones", color="Algoritmo",
                           facet_col="Sentido", barmode="group", log_y=True, height=330,
                           title="Trabajo de cada algoritmo (escala log)"), width="stretch")

    st.subheader("Ruta actual vs. alternativas por destino")
    destino = st.selectbox("Destino", DESTINOS)
    act = RUTAS_ACTUALES[destino]
    candidatos = {"Actual (hoja de despacho)": (act, list(reversed(act)))}
    ida_d, _, _ = dijkstra(G, BODEGA, destino)
    reg_d, _, _ = dijkstra(G, destino, BODEGA)
    candidatos["Dijkstra (ida y regreso óptimos)"] = (ida_d, reg_d)
    yen = [p for p in rutas_alternativas(G, BODEGA, destino, 4) if p != ida_d and p != act]
    if yen:
        candidatos["2.ª más corta (Yen), regreso invertido"] = (yen[0], list(reversed(yen[0])))
    bfs = ruta_menos_tramos(G, BODEGA, destino)
    if bfs and bfs not in (act, ida_d):
        candidatos["Menos intersecciones (BFS), regreso invertido"] = (bfs, list(reversed(bfs)))
    filas = []
    for nombre, (ida, reg) in candidatos.items():
        ti, tr = tiempo_ruta(G, ida), tiempo_ruta(G, reg)
        filas.append({"Alternativa": nombre, "Ida": " → ".join(ida), "Tiempo ida": ti,
                      "Regreso": " → ".join(reg), "Tiempo regreso": tr,
                      "Ciclo": None if ti is None or tr is None else ti + tr,
                      "Puntual sin espera": None if ti is None else ti + t_descarga <= LIMITE_PUNTUAL})
    df_c = pd.DataFrame(filas)
    st.dataframe(df_c, hide_index=True, width="stretch")
    st.caption("Tiempo vacío = la ruta usa un tramo inexistente o cerrado.")
    st.plotly_chart(figura_grafo(G, POS, {k: v[0] for k, v in candidatos.items()}, cerrados, False, 560),
                    width="stretch")

    st.subheader("Resumen: plan actual vs. plan Dijkstra")
    res = PLAN_ACT.tabla()[["Destino", "Tiempo ida", "Tiempo regreso", "Ciclo"]].merge(
        PLAN_OPT.tabla()[["Destino", "Tiempo ida", "Tiempo regreso", "Ciclo"]], on="Destino",
        suffixes=(" actual", " Dijkstra"))
    res["Ahorro por viaje"] = res["Ciclo actual"] - res["Ciclo Dijkstra"]
    st.dataframe(res, hide_index=True, width="stretch")
    ca, co = PLAN_ACT.tabla()["Ciclo"].mean(), PLAN_OPT.tabla()["Ciclo"].mean()
    st.info(f"Ciclo promedio (destinos equiprobables): actual **{ca:.1f} min**, Dijkstra **{co:.1f} min**. "
            f"Capacidad teórica con {vehiculos} vehículos: actual **{vehiculos * 60 / ca:.2f}** pedidos/h, "
            f"Dijkstra **{vehiculos * 60 / co:.2f}** pedidos/h (demanda: {tasa:g}/h).")

# ---------------------------------------------------------------- simulación
with tab_sim:
    pedidos = generar_pedidos(tasa, horas, int(semilla))
    st.write(f"Semilla **{semilla}** → **{len(pedidos)}** pedidos en {horas:g} h. "
             "Ambas políticas atienden exactamente los mismos pedidos.")
    if not (PLAN_ACT.factible and PLAN_OPT.factible):
        st.error("Algún destino quedó inaccesible con los incidentes actuales.")
    else:
        resultados = {n: simular(pedidos, p, vehiculos, horas, t_descarga)
                      for n, p in {"Rutas actuales": PLAN_ACT, "Rutas Dijkstra": PLAN_OPT}.items()}
        m = pd.DataFrame({n: r[1] for n, r in resultados.items()})
        cols = st.columns(2)
        for col, (nombre, (df, met)) in zip(cols, resultados.items()):
            with col:
                st.subheader(nombre)
                k = st.columns(3)
                k[0].metric("Puntuales", f"{met['Entregas puntuales']}/{met['Pedidos']}", f"{met['% puntual']:.1f}%",
                            delta_color="off")
                k[1].metric("Espera prom.", f"{met['Espera promedio']:.1f} min")
                k[2].metric("Solicitud→entrega", f"{met['Solicitud→entrega promedio']:.1f} min")
                k = st.columns(3)
                k[0].metric("Utilización", f"{met['Utilización']:.1f}%")
                k[1].metric("Tiempo adicional", f"{met['Tiempo adicional']:.0f} min")
                k[2].metric("Espera máx.", f"{met['Espera máxima']:.1f} min")
                st.plotly_chart(figura_cola(df, horas, "Cola y vehículos en ruta"), width="stretch")
                st.plotly_chart(figura_gantt(df, horas, "Ocupación de vehículos"), width="stretch")
        st.subheader("Métricas lado a lado")
        st.dataframe(m.round(2), width="stretch")
        both = pd.concat([r[0].assign(Política=n) for n, r in resultados.items()])
        st.plotly_chart(px.histogram(both, x="Solicitud→entrega", color="Política", barmode="overlay",
                                     nbins=40, height=330, title="Distribución del tiempo solicitud→entrega")
                        .add_vline(x=LIMITE_PUNTUAL, line_dash="dash", annotation_text="75 min"),
                        width="stretch")
        with st.expander("Detalle por pedido"):
            st.dataframe(both.round(2), hide_index=True, width="stretch")

# ---------------------------------------------------------------- experimentos
with tab_exp:
    st.write("Corre ambas políticas con las mismas semillas (mismos pedidos, destinos e instantes de "
             "llegada en cada pareja) para los tres escenarios y el personalizado si aplica, usando la "
             "flota, duración y red actuales.")
    c = st.columns(3)
    reps = c[0].number_input("Repeticiones por escenario", 5, 200, 10)
    semilla0 = c[1].number_input("Semilla inicial", 0, 1_000_000, int(semilla))
    correr = c[2].button("▶️ Ejecutar experimentos", type="primary", width="stretch")
    if correr:
        escenarios = dict(ESCENARIOS)
        if escenario == "Personalizado":
            escenarios["Personalizado"] = tasa
        semillas = list(range(int(semilla0), int(semilla0) + int(reps)))
        st.session_state.exp = pd.concat([comparar(G, t, vehiculos, horas, semillas, t_descarga, e)
                                          for e, t in escenarios.items()])
    if "exp" in st.session_state:
        ex = st.session_state.exp
        orden = list(ex["Escenario"].unique())
        resumen = ex.groupby(["Escenario", "Política"], sort=False).agg(["mean", "std"]).drop(columns="Semilla")
        st.subheader("Promedio ± desviación estándar")
        st.dataframe(resumen.round(1), width="stretch")
        for metrica in ["% puntual", "Espera promedio", "Solicitud→entrega promedio", "Utilización",
                        "Tiempo adicional"]:
            st.plotly_chart(px.box(ex, x="Escenario", y=metrica, color="Política", points="all", height=330,
                                   category_orders={"Escenario": orden}, title=metrica),
                            width="stretch")
        par = ex.pivot_table(index=["Escenario", "Semilla"], columns="Política",
                             values=["% puntual", "Espera promedio", "Tiempo adicional"])
        dif = pd.DataFrame({
            "Δ % puntual": par[("% puntual", "Rutas Dijkstra")] - par[("% puntual", "Rutas actuales")],
            "Δ espera (min)": par[("Espera promedio", "Rutas Dijkstra")] - par[("Espera promedio", "Rutas actuales")],
            "Δ tiempo adicional (min)": par[("Tiempo adicional", "Rutas Dijkstra")] - par[("Tiempo adicional", "Rutas actuales")],
        }).groupby("Escenario", sort=False).agg(["mean", "min", "max"])
        st.subheader("Diferencia pareada (Dijkstra − actual) por escenario")
        st.dataframe(dif.round(1), width="stretch")
        st.download_button("Descargar resultados (CSV)", ex.to_csv(index=False), "experimentos.csv")

    st.divider()
    st.subheader("¿Cuándo limita la flota?")
    st.write("Barrido del número de vehículos para el escenario seleccionado (promedio de 5 semillas).")
    if st.button("Calcular barrido de flota"):
        filas = [comparar(G, tasa, v, horas, list(range(int(semilla), int(semilla) + 5)), t_descarga, escenario)
                 .assign(Vehículos=v) for v in range(2, 25)]
        st.session_state.barrido = pd.concat(filas).groupby(["Vehículos", "Política"], as_index=False).mean(numeric_only=True)
    if "barrido" in st.session_state:
        b = st.session_state.barrido
        cc = st.columns(2)
        cc[0].plotly_chart(px.line(b, x="Vehículos", y="% puntual", color="Política", markers=True, height=350,
                                   title="% puntual vs. flota"), width="stretch")
        cc[1].plotly_chart(px.line(b, x="Vehículos", y="Espera promedio", color="Política", markers=True,
                                   height=350, title="Espera promedio vs. flota"), width="stretch")
