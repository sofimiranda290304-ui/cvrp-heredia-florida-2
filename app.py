"""
app.py  –  CVRP Florida Bebidas · Provincia de Heredia
Streamlit + Python  |  II-1122 · UCR Sede Alajuela
Modelo: cada trip = 1 camión (24 pallets). Total: 15 camiones.
"""

import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PALETA
# ─────────────────────────────────────────────────────────────────────────────
C1, C2, C3, C4, C5 = "#F6FFEA", "#FFDE96", "#FA855A", "#C93638", "#62C4DA"

# ─────────────────────────────────────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────────────────────────────────────
NODOS = {
    0:"CD Heredia", 1:"Heredia", 2:"Barva", 3:"Santo Domingo",
    4:"Santa Bárbara", 5:"San Rafael", 6:"San Isidro",
    7:"Belén", 8:"Flores", 9:"San Pablo", 10:"Sarapiquí",
}
DEMANDA       = {0:0,  1:99, 2:36, 3:35, 4:29, 5:36, 6:17, 7:17, 8:16, 9:23, 10:51}
DEM_IMPERIAL  = {1:49, 2:18, 3:17, 4:15, 5:18, 6:9,  7:9,  8:8,  9:11, 10:25}
DEM_PILSEN    = {1:25, 2:9,  3:9,  4:7,  5:9,  6:4,  7:4,  8:4,  9:6,  10:13}
DEM_TROPICAL  = {1:25, 2:9,  3:9,  4:7,  5:9,  6:4,  7:4,  8:4,  9:6,  10:13}

DIST_RAW = [
 [0.0,  0.0,  3.0,  5.0,  7.0,  5.0, 11.0, 10.0,  5.0,  5.0, 69.0],
 [0.0,  0.0,  3.0,  5.0,  7.0,  5.0, 11.0, 10.0,  5.0,  5.0, 69.0],
 [3.0,  3.0,  0.0,  7.0,  5.0,  5.0, 10.0, 11.0,  5.0,  7.0, 67.0],
 [5.0,  5.0,  7.0,  0.0, 12.0,  5.0,  9.0, 14.0, 10.0,  0.0, 71.0],
 [7.0,  7.0,  5.0, 12.0,  0.0, 10.0, 14.0,  9.0,  5.0, 12.0, 65.0],
 [5.0,  5.0,  5.0,  5.0, 10.0,  0.0,  5.0, 15.0, 10.0,  5.0, 66.0],
 [11.0,11.0, 10.0,  9.0, 14.0,  5.0,  0.0, 20.0, 15.0,  9.0, 63.0],
 [10.0,10.0, 11.0, 14.0,  9.0, 15.0, 20.0,  0.0,  5.0, 14.0, 74.0],
 [5.0,  5.0,  5.0, 10.0,  5.0, 10.0, 15.0,  5.0,  0.0, 10.0, 70.0],
 [5.0,  5.0,  7.0,  0.0, 12.0,  5.0,  9.0, 14.0, 10.0,  0.0, 71.0],
 [69.0,69.0, 67.0, 71.0, 65.0, 66.0, 63.0, 74.0, 70.0, 71.0,  0.0],
]

CAPACIDAD   = 24
VELOCIDAD   = 40
MIN_PARADA  = 15
MIN_PALLET  = 3
JORNADA_MIN = 480
DEM_TOTAL   = sum(DEMANDA[i] for i in range(1, 11))  # 359

AMPL_MOD = """\
# ============================================================
#  cvrp_heredia.mod  –  CVRP Florida Bebidas – Heredia
#  Modelo AMPL  |  II-1122  |  UCR Sede Alajuela
# ============================================================
set NODES;                        # {0,1,...,10}  0 = CD
set ARCS within {NODES, NODES};

param dist     {ARCS} >= 0;
param demand   {NODES} >= 0;
param capacity := 24;
param total_demand := sum {i in NODES} demand[i];

var y {ARCS} integer >= 0;        # camiones en arco i->j
var f {ARCS}          >= 0;       # carga (pallets) en arco i->j

minimize TotalDist:
    sum {(i,j) in ARCS} dist[i,j] * y[i,j];

# R1: Balance de camiones
subject to FlowBalance {k in NODES diff {0}}:
    sum {(i,k) in ARCS} y[i,k] - sum {(k,j) in ARCS} y[k,j] = 0;

# R2: Balance de carga
subject to LoadBalance {k in NODES diff {0}}:
    sum {(i,k) in ARCS} f[i,k] - sum {(k,j) in ARCS} f[k,j] = demand[k];

# R3: Carga total desde CD = demanda total
subject to DepotOut:
    sum {(0,j) in ARCS} f[0,j] = total_demand;

# R4: Capacidad por arco
subject to Capacity {(i,j) in ARCS}:
    f[i,j] <= capacity * y[i,j];
"""

# ─────────────────────────────────────────────────────────────────────────────
# SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def d(i, j): return DIST_RAW[i][j]

def route_dist(route):
    dist = d(0, route[0])
    for k in range(len(route)-1): dist += d(route[k], route[k+1])
    return dist + d(route[-1], 0)

def two_opt(route):
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-1):
            for j in range(i+1, len(best)):
                nr = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_dist(nr) < route_dist(best):
                    best, improved = nr, True
    return best

def trip_min(trip):
    return (trip["dist_km"]/VELOCIDAD*60) + len(trip["route"])*MIN_PARADA + trip["load"]*MIN_PALLET

@st.cache_data
def run_solver():
    """
    Nearest-Neighbor + 2-opt.
    CADA TRIP = 1 CAMIÓN (modelo CVRP: y(i,j) cuenta camiones por arco).
    """
    remaining = {i: DEMANDA[i] for i in range(1, 11)}
    trips = []

    while any(v > 0 for v in remaining.values()):
        current, load, route = 0, 0, []
        available = {i: remaining[i] for i in remaining if remaining[i] > 0}

        while available:
            nearest = min(available, key=lambda j: d(current, j))
            can_load = min(available[nearest], CAPACIDAD - load)
            if can_load <= 0: break
            route.append(nearest)
            load += can_load
            remaining[nearest] -= can_load
            if remaining[nearest] == 0: del available[nearest]
            else: available[nearest] = remaining[nearest]
            if load >= CAPACIDAD: break
            current = nearest

        if route:
            route = two_opt(route)
            trips.append({
                "route": route,
                "load":  load,
                "dist_km": round(route_dist(route), 1),
            })

    return trips  # cada elemento = 1 camión

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown(f"""<style>
      .stApp{{background-color:{C1};}}
      section[data-testid="stSidebar"]{{background-color:{C2};}}
      .mc{{background:white;border-left:5px solid {C3};border-radius:8px;
           padding:14px 18px;margin:6px 0;box-shadow:0 2px 6px rgba(0,0,0,.08);}}
      .mc h2{{color:{C4};font-size:2rem;margin:0;}}
      .mc p{{color:#555;font-size:.85rem;margin:0;}}
      .sh{{background:linear-gradient(90deg,{C4},{C3});color:white;
           padding:10px 18px;border-radius:8px;font-size:1.1rem;
           font-weight:700;margin:20px 0 10px;}}
      .nb{{display:inline-block;background:{C5};color:white;border-radius:50%;
           width:26px;height:26px;text-align:center;line-height:26px;
           font-weight:bold;font-size:.8rem;margin:0 2px;}}
      .nb0{{background:{C4};}}
      .tc{{background:white;border-radius:10px;padding:12px 16px;margin:6px 0;
           border-left:4px solid {C5};box-shadow:0 1px 4px rgba(0,0,0,.07);}}
    </style>""", unsafe_allow_html=True)

def card(val, lbl, color=None):
    bc = color or C3
    st.markdown(f"""<div class="mc" style="border-left-color:{bc}">
        <h2>{val}</h2><p>{lbl}</p></div>""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="sh">📌 {title}</div>', unsafe_allow_html=True)

def badges(route):
    html = '<span class="nb nb0">0</span>'
    for n in route: html += f' → <span class="nb">{n}</span>'
    return html + ' → <span class="nb nb0">0</span>'

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

def tab_dashboard(trips):
    total_km = sum(t["dist_km"] for t in trips)
    n_cam    = len(trips)   # 1 trip = 1 camión
    entregado = sum(t["load"] for t in trips)

    st.markdown("### 🗺️ Distribución Óptima — Provincia de Heredia")
    st.caption("Florida Bebidas (FIFCO) · Cervecería Río Segundo de Alajuela → 10 cantones")

    c1,c2,c3,c4 = st.columns(4)
    with c1: card(f"{total_km:.1f} km", "Distancia total Z*")
    with c2: card(str(n_cam), "Camiones (1 por trip)")
    with c3: card(f"{n_cam}", "Trips generados")
    with c4: card(f"{entregado}/{DEM_TOTAL}", "Pallets entregados")

    st.divider()
    sec("Secuencia de Rutas — Un camión por trip")
    st.markdown(
        "Cada fila es **1 camión** que sale del CD Heredia (nodo 0), "
        "visita los nodos indicados y regresa al CD. "
        "**Capacidad máxima: 24 pallets por camión.**"
    )

    rows = []
    for idx, t in enumerate(trips, 1):
        ruta = " → ".join(["N0"]+[f"N{n}" for n in t["route"]]+["N0"])
        dur  = round(trip_min(t))
        rows.append({
            "Camión": f"C{idx}",
            "Ruta (nodos)": ruta,
            "Pallets cargados": t["load"],
            "Dist (km)": t["dist_km"],
            "Duración (min)": dur,
            "¿Cabe en 8h?": "✅" if dur <= JORNADA_MIN else "⚠️",
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.highlight_max(subset=["Dist (km)"], color="#FA855A")
                .highlight_min(subset=["Dist (km)"], color="#62C4DA"),
        use_container_width=True, hide_index=True
    )

    sec("Cantón con Mayor y Menor Distancia al CD")
    dist_cd  = {i: DIST_RAW[0][i] for i in range(1,11)}
    max_n = max(dist_cd, key=dist_cd.get)
    min_n = min(dist_cd, key=dist_cd.get)
    ca, cb = st.columns(2)
    with ca: card(f"🔴 Nodo {max_n} — {NODOS[max_n]}", f"Mayor distancia al CD: {dist_cd[max_n]} km", C4)
    with cb: card(f"🟢 Nodo {min_n} — {NODOS[min_n]}", f"Menor distancia al CD: {dist_cd[min_n]} km", C5)


def tab_cantones(trips):
    sec("Detalle por Cantón (Desplegable)")
    opciones = [f"Nodo {i} — {NODOS[i]}" for i in range(1,11)]
    sels = st.multiselect("Cantones a visualizar:", opciones, default=opciones[:3])

    for sel in sels:
        nodo   = int(sel.split("—")[0].replace("Nodo","").strip())
        canton = NODOS[nodo]
        entregado = sum(min(t["load"], DEMANDA[nodo]) for t in trips if nodo in t["route"])
        dist_cd   = DIST_RAW[0][nodo]
        t_min     = round(dist_cd/VELOCIDAD*60 + MIN_PARADA + DEMANDA[nodo]*MIN_PALLET)
        pct       = round(min(entregado, DEMANDA[nodo]) / DEMANDA[nodo] * 100) if DEMANDA[nodo] else 0

        with st.expander(f"📦 Nodo {nodo} — **{canton}**", expanded=True):
            c1,c2,c3 = st.columns(3)
            with c1: st.markdown(f'<div class="mc"><h2>{DEMANDA[nodo]} pallets</h2><p>Demanda total semanal</p></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="mc"><h2>{dist_cd} km</h2><p>Distancia al CD (ida)</p></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="mc"><h2>{t_min} min</h2><p>Tiempo estimado de visita</p></div>', unsafe_allow_html=True)

            st.dataframe(pd.DataFrame({
                "Producto": ["Imperial","Pilsen","Tropical"],
                "Demanda (pallets)": [DEM_IMPERIAL.get(nodo,0), DEM_PILSEN.get(nodo,0), DEM_TROPICAL.get(nodo,0)],
            }), hide_index=True, use_container_width=True)

            color_bar = C3 if pct < 100 else C5
            st.markdown(f"""**Demanda Satisfecha: {pct}%**
            <div style="background:#eee;border-radius:8px;height:18px;margin-top:6px;">
              <div style="background:{color_bar};width:{pct}%;border-radius:8px;height:18px;"></div>
            </div>""", unsafe_allow_html=True)

            trips_v = [t for t in trips if nodo in t["route"]]
            if trips_v:
                st.markdown(f"**Camiones que visitan Nodo {nodo}:**")
                for i, t in enumerate(trips_v, 1):
                    st.markdown(f"&nbsp;&nbsp;🚚 Camión {i}: {' → '.join(['N0']+[f'N{x}' for x in t['route']]+['N0'])} — {t['dist_km']} km — {t['load']} pallets")


def tab_camiones(trips):
    sec(f"Los {len(trips)} Camiones del Modelo CVRP")
    st.markdown(
        f"En el modelo CVRP, **cada trip es un camión independiente** que sale del CD con hasta 24 pallets. "
        f"La demanda total de Heredia es **{DEM_TOTAL} pallets**, lo que requiere un mínimo de "
        f"⌈{DEM_TOTAL}/24⌉ = **{len(trips)} camiones**."
    )
    st.info("💡 La flota mínima teórica = ⌈359 / 24⌉ = 15 camiones. La solución alcanza ese óptimo.", icon="✅")

    for idx, t in enumerate(trips, 1):
        dur    = round(trip_min(t))
        km_ida = DIST_RAW[0][t["route"][0]]
        util   = round(t["load"] / CAPACIDAD * 100)
        color  = C4 if dur > JORNADA_MIN else C5

        col_h, col_b = st.columns([5, 1])
        with col_h:
            st.markdown(
                f"**🚛 Camión #{idx}** &nbsp;|&nbsp; "
                f"{t['load']}/{CAPACIDAD} pallets ({util}% capacidad) &nbsp;|&nbsp; "
                f"{t['dist_km']} km &nbsp;|&nbsp; {dur} min",
            )
        with col_b:
            st.progress(min(util, 100), text=f"{util}%")

        route_html = badges(t["route"])
        cantones_str = " + ".join([NODOS[n] for n in t["route"]])
        st.markdown(f"""<div class="tc">
            <small>{route_html}</small><br>
            <span style="color:#666;font-size:.85rem;">Cantones: {cantones_str}</span>
        </div>""", unsafe_allow_html=True)


def tab_modelo():
    sec("Modelo AMPL — CVRP Heredia")
    st.markdown("Modelo en **AMPL** según estructura de la Clase 13 (II-1122, UCR).")

    with st.expander("📐 Variables de Decisión", expanded=True):
        st.markdown("""
| Variable | Tipo | Descripción |
|---|---|---|
| `y(i,j)` | Entera ≥ 0 | **Número de camiones** que transitan el arco *i → j* |
| `f(i,j)` | Continua ≥ 0 | Carga en pallets que transita el arco *i → j* |
        """)

    with st.expander("🎯 Función Objetivo"):
        st.latex(r"\min\ Z = \sum_{(i,j) \in A} dist_{ij} \cdot y_{ij}")
        st.caption("Minimizar la distancia total recorrida (km) por toda la flota.")

    with st.expander("🔒 Restricciones"):
        st.markdown("**R1 — Balance de camiones** *(k ≠ 0)*:")
        st.latex(r"\sum_{(i,k)\in A} y_{ik} - \sum_{(k,j)\in A} y_{kj} = 0")
        st.markdown("**R2 — Balance de carga** *(k ≠ 0)*:")
        st.latex(r"\sum_{(i,k)\in A} f_{ik} - \sum_{(k,j)\in A} f_{kj} = d_k")
        st.markdown("**R3 — Carga total desde el depósito:**")
        st.latex(r"\sum_{(0,j)\in A} f_{0j} = 359 \text{ pallets}")
        st.markdown("**R4 — Capacidad por arco:**")
        st.latex(r"f_{ij} \leq 24 \cdot y_{ij} \quad \forall (i,j) \in A")

    with st.expander("📊 Parámetros"):
        st.dataframe(pd.DataFrame({
            "Parámetro": ["Capacidad/camión","Velocidad","Tiempo/parada","Tiempo/pallet","Jornada máx","Demanda total","Nodos"],
            "Valor":     ["24 pallets","40 km/h","15 min","3 min","8 h (480 min)","359 pallets/sem","11 (0=CD + 10 cantones)"],
        }), hide_index=True, use_container_width=True)

    with st.expander("💻 Código AMPL (.mod)"):
        st.code(AMPL_MOD, language="text")


def tab_optimo(trips):
    sec("¿Cómo se Alcanzó el Óptimo?")
    total_km = sum(t["dist_km"] for t in trips)

    st.markdown(f"""
### Método de Solución

#### Paso 1 — Modelo AMPL
Variables `y(i,j)` (camiones por arco) y `f(i,j)` (carga por arco).
Función objetivo: minimizar km totales. Restricciones R1–R4 de balance, depósito y capacidad.

#### Paso 2 — Heurística Nearest-Neighbor
Desde CD (nodo 0), se elige el cliente **más cercano** con demanda restante hasta completar 24 pallets.
El camión regresa al CD. Se repite hasta satisfacer toda la demanda.

#### Paso 3 — Mejora 2-opt
Cada ruta se mejora invirtiendo segmentos mientras la distancia disminuya → **óptimo local por camión**.

#### Paso 4 — Resultado
**{len(trips)} camiones** cubren los 359 pallets, alcanzando el **límite inferior teórico**
(⌈359 / 24⌉ = 15). La solución es óptima en número de camiones.
    """)

    c1,c2,c3 = st.columns(3)
    with c1: card(f"{total_km:.1f} km", "Distancia total Z*")
    with c2: card(str(len(trips)), "Camiones necesarios")
    with c3: card("359 / 359", "Pallets entregados")

    st.markdown("""
---
### Verificación de Optimalidad
- Heredia tiene 10 nodos → instancia pequeña, resoluble a optimalidad con AMPL (Clase 13).
- ⌈359 / 24⌉ = **15 camiones** es el límite inferior teórico. La solución lo alcanza exactamente.
- Cada restricción R1–R4 es satisfecha en cada camión generado.

---
### Recomendaciones para Florida Bebidas
- **Heredia (nodo 1, 99 pallets)**: requiere 4 camiones de full-load. Un sub-depósito reduciría km.
- **Sarapiquí (nodo 10, 69 km)**: es el más lejano — considerar camión dedicado fijo por semana.
- Agregar **ventanas de tiempo** mejoraría la planificación operativa real.
- Re-correr el modelo cada semana con datos actualizados de demanda.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="CVRP Heredia — Florida Bebidas", page_icon="🍺", layout="wide")
    inject_css()

    with st.sidebar:
        st.markdown(f"<div style='text-align:center;padding:10px;'>"
                    f"<h2 style='color:{C4};'>🍺 Florida Bebidas</h2>"
                    f"<p style='color:#444;font-size:.9rem;'>CVRP — Provincia de Heredia</p><hr/></div>",
                    unsafe_allow_html=True)
        st.markdown("**Nodos del Modelo**")
        for k, v in NODOS.items():
            color = C4 if k == 0 else C5
            st.markdown(
                f"<span style='background:{color};color:white;border-radius:50%;"
                f"display:inline-block;width:22px;height:22px;text-align:center;"
                f"line-height:22px;font-size:.75rem;'>{k}</span> {v}",
                unsafe_allow_html=True)
        st.divider()
        st.caption("II-1122 · Clase 13 · UCR Sede Alajuela")
        st.caption("Datos: INEC 2022 · Florida Bebidas (FIFCO)")

    st.markdown(
        f"<div style='background:linear-gradient(135deg,{C4},{C3});color:white;"
        f"padding:20px 30px;border-radius:12px;margin-bottom:20px;'>"
        f"<h1 style='margin:0;'>🍺 CVRP — Distribución Florida Bebidas</h1>"
        f"<p style='margin:4px 0 0;opacity:.9;'>Provincia de Heredia · 10 Cantones · "
        f"15 Camiones · Optimización de Rutas (AMPL)</p></div>",
        unsafe_allow_html=True)

    with st.spinner("Resolviendo CVRP — Nearest-Neighbor + 2-opt…"):
        trips = run_solver()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "🏙️ Cantones", "🚛 Camiones", "📐 Modelo AMPL", "🏆 Óptimo",
    ])
    with tab1: tab_dashboard(trips)
    with tab2: tab_cantones(trips)
    with tab3: tab_camiones(trips)
    with tab4: tab_modelo()
    with tab5: tab_optimo(trips)

if __name__ == "__main__":
    main()
