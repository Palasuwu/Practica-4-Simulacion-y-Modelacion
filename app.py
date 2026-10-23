"""Practica 4 Matrices: estudio de rutas y flota de Logística del Valle. Ejecutar con: streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from algoritmos import ALGORITMOS, alternativas_destino
from grafo import (BODEGA, DESTINOS, cargar_matriz, construir_grafo, resumen_sentidos, tiempo_ruta,
                   validar_rutas)
from simulacion import (ESCENARIOS, LIMITE_PUNTUAL, comparar, generar_pedidos, plan_actual,
                        plan_optimo, simular)
from visual import COLORES_POLITICA, PLANTILLA, figura_cola, figura_gantt, figura_grafo, posiciones

st.set_page_config(page_title="Practica 4 Matrices", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  h1 {font-weight: 600; letter-spacing: -0.01em;}
  h2, h3 {font-weight: 600;}
  [data-testid="stSidebar"] h2 {font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.04em;}
</style>
""", unsafe_allow_html=True)


def grafica(fig, **kw):
    fig.update_layout(template=PLANTILLA)
    st.plotly_chart(fig, width="stretch", **kw)


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
st.sidebar.header("Parámetros")
escenario = st.sidebar.radio("Escenario", [*ESCENARIOS, "Personalizado"])
tasa = st.sidebar.number_input("Pedidos por hora (promedio)", 0.5, 40.0,
                               ESCENARIOS.get(escenario, 4.0), 0.5, disabled=escenario != "Personalizado")
vehiculos = st.sidebar.slider("Número de vehículos", 1, 30, 8)
horas = st.sidebar.slider("Duración de la recepción (horas)", 1.0, 16.0, 8.0, 0.5)
semilla = st.sidebar.number_input("Semilla aleatoria", 0, 10_000_000, 42, 1)
t_descarga = st.sidebar.number_input("Tiempo de entrega en destino (min)", 0.0, 30.0, 0.0, 1.0,
                                     help="Tiempo de descarga; 0 = solo se cuenta el viaje.")

st.sidebar.divider()
st.sidebar.header("Incidentes viales")
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
if st.session_state.incidentes:
    st.sidebar.dataframe(pd.DataFrame([
        {"Tramo": f"{a}→{b}", "Estado": "cerrado" if t is None else f"{g_base[a][b]['weight']:g} → {t:g} min"}
        for (a, b), t in st.session_state.incidentes.items()]), hide_index=True, width="stretch")

G = red_actual()
cerrados = [k for k, v in st.session_state.incidentes.items() if v is None]
PLAN_ACT, PLAN_OPT = plan_actual(G), plan_optimo(G)

# ---------------------------------------------------------------- encabezado
st.title("Practica 4 Matrices")
st.caption(f"Bodega {BODEGA}  |  Destinos {', '.join(DESTINOS)}  |  Entrega puntual: {LIMITE_PUNTUAL:g} min o menos "
           f"desde la solicitud  |  Escenario: {escenario} ({tasa:g} pedidos/h)")
if st.session_state.incidentes:
    st.caption(f"Red modificada ({len(st.session_state.incidentes)} tramo(s)). Las rutas Dijkstra se recalculan "
               "sobre la red modificada; las rutas actuales toman el desvío más corto en cada tramo cerrado.")

tab_red, tab_rutas, tab_sim, tab_exp = st.tabs(["Red vial", "Rutas", "Simulación", "Experimentos"])

# ---------------------------------------------------------------- red
with tab_red:
    col_g, col_i = st.columns([3, 1])
    with col_i:
        resaltar = st.radio("Resaltar", ["Ninguna", "Rutas actuales (ida)", "Rutas Dijkstra (ida)",
                                         "Rutas Dijkstra (regreso)"])
        pesos = st.checkbox("Mostrar tiempos", True)
        r = resumen_sentidos(G)
        st.markdown(
            "| Elemento | Cantidad |\n|---|--:|\n"
            f"| Nodos | {G.number_of_nodes()} |\n"
            f"| Arcos dirigidos | {r['arcos']} |\n"
            f"| Doble sentido, mismo tiempo | {r['pares_doble_sentido_simetricos']} pares |\n"
            f"| Doble sentido, tiempo distinto | {r['pares_doble_sentido_asimetricos']} pares |\n"
            f"| Un solo sentido | {len(r['arcos_un_sentido'])} arcos |")
        st.caption("Gris: doble sentido (si el tiempo difiere se muestran ambos). Naranja con flecha: "
                   "un solo sentido. Negro: bodega. Rojo: destinos.")
    rutas = {"Rutas actuales (ida)": {f"{d} actual": PLAN_ACT.ida[d] for d in DESTINOS},
             "Rutas Dijkstra (ida)": {f"{d} Dijkstra": PLAN_OPT.ida[d] for d in DESTINOS},
             "Rutas Dijkstra (regreso)": {f"{d} regreso": PLAN_OPT.regreso[d] for d in DESTINOS}}.get(resaltar)
    with col_g:
        grafica(figura_grafo(G, POS, rutas, cerrados, pesos))

    st.subheader("Validación de las rutas registradas")
    st.caption("Ida según las hojas de despacho; regreso por el mismo camino en sentido inverso.")
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
                              "Nodos expandidos / relajaciones": trabajo, "Ruta": " → ".join(ruta) or "sin ruta"})
    df_alg = pd.DataFrame(filas)
    st.dataframe(df_alg, hide_index=True, width="stretch")
    grafica(px.bar(df_alg, x="Destino", y="Nodos expandidos / relajaciones", color="Algoritmo",
                   facet_col="Sentido", barmode="group", log_y=True, height=330,
                   color_discrete_sequence=["#1f3a5f", "#6b8fb5", "#b8b8b8"],
                   title="Trabajo de cada algoritmo (escala logarítmica)"))

    st.subheader("Ruta actual y alternativas por destino")
    destino = st.selectbox("Destino", DESTINOS)
    candidatos = {n: (ida, reg) for n, ida, reg in alternativas_destino(G, destino)}
    filas = []
    for nombre, (ida, reg) in candidatos.items():
        ti, tr = tiempo_ruta(G, ida), tiempo_ruta(G, reg)
        filas.append({"Alternativa": nombre, "Ida": " → ".join(ida), "Tiempo ida": ti,
                      "Regreso": " → ".join(reg), "Tiempo regreso": tr,
                      "Ciclo": None if ti is None or tr is None else ti + tr,
                      "Ida invertida válida": tiempo_ruta(G, ida[::-1]) is not None,
                      "Puntual sin espera": None if ti is None else ti + t_descarga <= LIMITE_PUNTUAL})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
    st.caption("Las alternativas calculan el regreso por separado, porque varias idas no se pueden recorrer "
               "al revés (calles de un solo sentido). Tiempo vacío: la ruta usa un tramo cerrado.")
    grafica(figura_grafo(G, POS, {k: v[0] for k, v in candidatos.items()}, cerrados, False, 560))

    st.subheader("Plan actual y plan Dijkstra")
    res = PLAN_ACT.tabla()[["Destino", "Tiempo ida", "Tiempo regreso", "Ciclo"]].merge(
        PLAN_OPT.tabla()[["Destino", "Tiempo ida", "Tiempo regreso", "Ciclo"]], on="Destino",
        suffixes=(" actual", " Dijkstra"))
    res["Ahorro por viaje"] = res["Ciclo actual"] - res["Ciclo Dijkstra"]
    st.dataframe(res, hide_index=True, width="stretch")
    ca, co = PLAN_ACT.tabla()["Ciclo"].mean(), PLAN_OPT.tabla()["Ciclo"].mean()
    st.dataframe(pd.DataFrame({
        "Plan": ["Actual", "Dijkstra"],
        "Ciclo promedio (min)": [round(ca, 1), round(co, 1)],
        f"Capacidad con {vehiculos} vehículos (pedidos/h)": [round(vehiculos * 60 / ca, 2), round(vehiculos * 60 / co, 2)],
        "Demanda (pedidos/h)": [tasa, tasa],
    }), hide_index=True)

# ---------------------------------------------------------------- simulación
with tab_sim:
    pedidos = generar_pedidos(tasa, horas, int(semilla))
    st.write(f"Semilla {semilla}: {len(pedidos)} pedidos en {horas:g} h. "
             "Ambas políticas atienden exactamente los mismos pedidos.")
    if not (PLAN_ACT.factible and PLAN_OPT.factible):
        st.error("Algún destino quedó inaccesible con los incidentes actuales.")
    else:
        resultados = {n: simular(pedidos, p, vehiculos, horas, t_descarga)
                      for n, p in {"Rutas actuales": PLAN_ACT, "Rutas Dijkstra": PLAN_OPT}.items()}
        st.subheader("Indicadores")
        tabla = pd.DataFrame({n: {
            "Entregas puntuales": f"{m['Entregas puntuales']} de {m['Pedidos']} ({m['% puntual']:.1f} %)",
            "Espera promedio (min)": f"{m['Espera promedio']:.1f}",
            "Espera máxima (min)": f"{m['Espera máxima']:.1f}",
            "Solicitud → entrega promedio (min)": f"{m['Solicitud→entrega promedio']:.1f}",
            "Utilización de vehículos (%)": f"{m['Utilización']:.1f}",
            "Tiempo adicional tras el cierre (min)": f"{m['Tiempo adicional']:.0f}",
        } for n, (_, m) in resultados.items()})
        st.dataframe(tabla, width="stretch")

        cols = st.columns(2)
        for col, (nombre, (df, _)) in zip(cols, resultados.items()):
            with col:
                grafica(figura_cola(df, horas, f"{nombre}: cola y vehículos en ruta"))
        st.subheader("Ocupación de vehículos")
        cols = st.columns(2)
        for col, (nombre, (df, _)) in zip(cols, resultados.items()):
            with col:
                grafica(figura_gantt(df, horas, nombre))
        both = pd.concat([r[0].assign(Política=n) for n, r in resultados.items()])
        grafica(px.histogram(both, x="Solicitud→entrega", color="Política", barmode="overlay", nbins=40,
                             height=330, color_discrete_map=COLORES_POLITICA,
                             title="Distribución del tiempo solicitud → entrega")
                .add_vline(x=LIMITE_PUNTUAL, line_dash="dash", annotation_text="75 min"))
        with st.expander("Detalle por pedido"):
            st.dataframe(both.round(2), hide_index=True, width="stretch")

# ---------------------------------------------------------------- experimentos
with tab_exp:
    st.write("Ejecuta ambas políticas con las mismas semillas (mismos pedidos, destinos e instantes de "
             "llegada en cada pareja) para los tres escenarios, y el personalizado si está seleccionado, "
             "con la flota, la duración y la red actuales.")
    c = st.columns([1, 1, 1])
    reps = c[0].number_input("Repeticiones por escenario", 5, 200, 10)
    semilla0 = c[1].number_input("Semilla inicial", 0, 1_000_000, int(semilla))
    c[2].write("")
    correr = c[2].button("Ejecutar experimentos", type="primary", width="stretch")
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
        st.subheader("Promedio y desviación estándar")
        st.dataframe(resumen.round(1), width="stretch")
        metricas = ["% puntual", "Espera promedio", "Solicitud→entrega promedio", "Utilización", "Tiempo adicional"]
        cols = st.columns(2)
        for i, metrica in enumerate(metricas):
            with cols[i % 2]:
                grafica(px.box(ex, x="Escenario", y=metrica, color="Política", points="all", height=320,
                               category_orders={"Escenario": orden}, color_discrete_map=COLORES_POLITICA,
                               title=metrica))
        par = ex.pivot_table(index=["Escenario", "Semilla"], columns="Política",
                             values=["% puntual", "Espera promedio", "Tiempo adicional"])
        dif = pd.DataFrame({
            "Δ % puntual": par[("% puntual", "Rutas Dijkstra")] - par[("% puntual", "Rutas actuales")],
            "Δ espera (min)": par[("Espera promedio", "Rutas Dijkstra")] - par[("Espera promedio", "Rutas actuales")],
            "Δ tiempo adicional (min)": par[("Tiempo adicional", "Rutas Dijkstra")] - par[("Tiempo adicional", "Rutas actuales")],
        }).groupby("Escenario", sort=False).agg(["mean", "min", "max"])
        st.subheader("Diferencia pareada por escenario (Dijkstra menos actual)")
        st.dataframe(dif.round(1), width="stretch")
        st.download_button("Descargar resultados (CSV)", ex.to_csv(index=False), "experimentos.csv")

    st.divider()
    st.subheader("Tamaño de la flota")
    st.write("Barrido del número de vehículos para el escenario seleccionado (promedio de 5 semillas).")
    if st.button("Calcular barrido de flota"):
        filas = [comparar(G, tasa, v, horas, list(range(int(semilla), int(semilla) + 5)), t_descarga, escenario)
                 .assign(Vehículos=v) for v in range(2, 25)]
        st.session_state.barrido = pd.concat(filas).groupby(["Vehículos", "Política"], as_index=False).mean(numeric_only=True)
    if "barrido" in st.session_state:
        b = st.session_state.barrido
        cc = st.columns(2)
        with cc[0]:
            grafica(px.line(b, x="Vehículos", y="% puntual", color="Política", markers=True, height=350,
                            color_discrete_map=COLORES_POLITICA, title="Entregas puntuales (%) según la flota"))
        with cc[1]:
            grafica(px.line(b, x="Vehículos", y="Espera promedio", color="Política", markers=True, height=350,
                            color_discrete_map=COLORES_POLITICA, title="Espera promedio (min) según la flota"))
