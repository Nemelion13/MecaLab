from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.typing import UploadedFile


st.set_page_config(
    page_title="MecaLab | Calculs mécaniques",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; color: white;}
    .hero p {margin: .45rem 0 0; color: #d9f0f5;}
    div[data-testid="stMetric"] {
        background: #f3f8fb;
        border: 1px solid #dce8ef;
        padding: 12px 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_uploaded_csv(
    uploaded_file: UploadedFile,
    delimiter: str,
    encoding: str,
    decimal_separator: str,
) -> pd.DataFrame:
    """Read a user CSV while allowing common separators and legacy encodings."""
    separator = None if delimiter == "Automatique" else delimiter
    options: dict[str, object] = {
        "comment": "#",
        "encoding": encoding,
        "decimal": decimal_separator,
    }
    if separator is None:
        return pd.read_csv(BytesIO(uploaded_file.getvalue()), sep=None, engine="python", **options)
    return pd.read_csv(BytesIO(uploaded_file.getvalue()), sep=separator, **options)


def numeric_column(data: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(data[column], errors="coerce")


def calculate_offset_yield(
    strain: np.ndarray,
    stress: np.ndarray,
    young_modulus: float,
    intercept: float,
    offset: float = 0.002,
) -> tuple[float, float] | None:
    """Return the first measured intersection with the parallel offset line."""
    direction = 1.0 if abs(np.nanmax(strain)) >= abs(np.nanmin(strain)) else -1.0
    valid = np.isfinite(strain) & np.isfinite(stress) & (direction * strain >= offset)
    x = strain[valid]
    y = stress[valid]
    if len(x) < 2:
        return None

    order = np.argsort(direction * x)
    x = x[order]
    y = y[order]
    difference = y - (young_modulus * (x - direction * offset) + intercept)
    crossings = np.flatnonzero(difference[:-1] * difference[1:] <= 0)
    if not len(crossings):
        return None

    index = int(crossings[0])
    delta = difference[index + 1] - difference[index]
    fraction = 0.0 if delta == 0 else -difference[index] / delta
    intersection_strain = x[index] + fraction * (x[index + 1] - x[index])
    intersection_stress = y[index] + fraction * (y[index + 1] - y[index])
    return float(intersection_strain), float(intersection_stress)


def show_plot_downloads(figure: go.Figure, filename: str, key: str) -> None:
    st.plotly_chart(
        figure,
        width="stretch",
        alt="Courbe mécanique interactive permettant de comparer les séries et d'observer leurs tendances.",
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": filename,
                "height": 800,
                "width": 1400,
                "scale": 2,
            },
        },
        key=key,
    )
    st.caption(
        "Zoom : sélectionnez une zone à la souris (double-clic pour réinitialiser). "
        "L'icône caméra de la barre du graphique permet d'enregistrer une image PNG."
    )
    st.download_button(
        "Télécharger le graphique interactif (HTML)",
        data=figure.to_html(full_html=True, include_plotlyjs=True).encode("utf-8"),
        file_name=f"{filename}.html",
        mime="text/html",
        key=f"{key}_html",
    )


def show_quick_calculator() -> None:
    st.subheader("Calculatrice rapide")
    st.caption(f"Calcul sur une éprouvette rectangulaire : $\\sigma$ = F / (largeur x épaisseur).")
    columns = st.columns(4)
    force = columns[0].number_input("Force (N)", value=1000.0, step=100.0, key="quick_force")
    width = columns[1].number_input(
        "Largeur (mm)", min_value=0.000001, value=10.0, step=0.5, key="quick_width"
    )
    thickness = columns[2].number_input(
        "Épaisseur (mm)", min_value=0.000001, value=2.0, step=0.1, key="quick_thickness"
    )
    length = columns[3].number_input(
        "Longueur initiale L₀ (mm)", min_value=0.000001, value=50.0, step=1.0,
        key="quick_length",
    )

    displacement = st.number_input(
        "Déplacement (mm)", value=0.1, step=0.01, format="%.5f", key="quick_displacement"
    )
    stress = force / (width * thickness)
    strain = displacement / length
    results = st.columns(3)
    results[0].metric("Section initiale", f"{width * thickness:.4g} mm²")
    results[1].metric("Contrainte nominale", f"{stress:.4g} MPa")
    results[2].metric("Déformation", f"{strain:.4%}")
    if strain != 0:
        st.info(f"Module sécant au point saisi : **{stress / strain:.4g} MPa**.")


def show_tensile_analysis() -> None:
    st.header("Calculs simples et traction / compression")
    show_quick_calculator()
    st.divider()

    st.subheader("Analyser un ou plusieurs essais")
    uploaded_files = st.file_uploader(
        "Déposez un ou plusieurs fichiers CSV",
        type=["csv"],
        accept_multiple_files=True,
        help="Les colonnes numériques des fichiers seront sélectionnées individuellement.",
        key="tensile_files",
    )
    if not uploaded_files:
        st.info("Chargez un fichier CSV pour tracer et analyser vos courbes.")
        return

    delimiter = st.selectbox(
        "Séparateur CSV", ["Automatique", ";", ",", "Tabulation"], key="csv_delimiter"
    )
    delimiter = "\t" if delimiter == "Tabulation" else delimiter

    parse_options = st.columns(2)
    encoding = parse_options[0].selectbox(
        "Encodage des CSV", ["utf-8", "latin1"], key="csv_encoding"
    )
    decimal_separator = parse_options[1].selectbox(
        "Séparateur décimal", [".", ","], key="decimal_separator"
    )

    analysis_columns = st.columns(2)
    strain_min = analysis_columns[0].number_input(
        "Déformation min. de la zone élastique",
        value=0.0,
        format="%.6f",
        key="elastic_min",
        help="Saisie en déformation sans dimension, même si le fichier donne des pourcentages.",
    )
    strain_max = analysis_columns[1].number_input(
        "Déformation max. de la zone élastique",
        value=0.001,
        format="%.6f",
        key="elastic_max",
    )
    if strain_max <= strain_min:
        st.error("La borne maximale de la zone élastique doit être supérieure à la borne minimale.")
        return

    plot_options = st.columns(2)
    curve_title = plot_options[0].text_input(
        "Titre du graphique", "Courbes contrainte - déformation", key="tensile_title"
    )
    show_offset = plot_options[1].checkbox(
        "Afficher la droite décalée Rp0,2", value=True, key="show_offset"
    )
    x_label = st.text_input("Titre de l'axe des abscisses", "Déformation ε", key="tensile_xlabel")
    y_label = st.text_input("Titre de l'axe des ordonnées", "Contrainte σ (MPa)", key="tensile_ylabel")
    zoom_elastic = st.checkbox(
        "Zoomer automatiquement sur la zone élastique", value=False, key="zoom_elastic"
    )
    with st.expander("Limites manuelles des axes"):
        manual_limits = st.checkbox("Activer les limites X/Y manuelles", key="manual_limits")
        limit_cols = st.columns(4)
        x_min = limit_cols[0].number_input("X min", value=0.0, format="%.6f", key="x_min")
        x_max = limit_cols[1].number_input("X max", value=0.01, format="%.6f", key="x_max")
        y_min = limit_cols[2].number_input("Y min (MPa)", value=0.0, key="y_min")
        y_max = limit_cols[3].number_input("Y max (MPa)", value=500.0, key="y_max")

    if zoom_elastic and manual_limits:
        st.warning("Choisissez soit le zoom élastique, soit les limites manuelles.")

    figure = go.Figure()
    metrics: list[dict[str, object]] = []
    palette = ["#176B87", "#E76F51", "#2A9D8F", "#7B61A8", "#E9A03B", "#344E41"]
    elastic_min_for_zoom: list[float] = []
    elastic_max_for_zoom: list[float] = []
    stress_factors = {
        "MPa (N/mm²)": 1.0,
        "Pa": 1e-6,
        "kPa": 1e-3,
        "GPa": 1e3,
    }

    curve_configs: list[dict[str, object]] = []
    for index, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"#### {uploaded_file.name}")
        try:
            data = read_uploaded_csv(uploaded_file, delimiter, encoding, decimal_separator)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            st.error(f"Impossible de lire **{uploaded_file.name}** : {error}")
            continue

        if data.empty or len(data.columns) == 0:
            st.warning(f"Le fichier **{uploaded_file.name}** ne contient pas de données.")
            continue

        options = data.columns.tolist()
        if len(options) < 2:
            st.warning(f"Le fichier **{uploaded_file.name}** doit contenir au moins deux colonnes.")
            continue

        with st.expander(f"Courbes et colonnes — {uploaded_file.name}", expanded=True):
            curve_count = st.number_input(
                "Nombre de courbes à extraire de ce fichier",
                min_value=1,
                max_value=12,
                value=1,
                step=1,
                key=f"curve_count_{index}",
                help="Chaque courbe utilise sa propre paire de colonnes et ses propres unités.",
            )
            st.dataframe(
                data.head(),
                width="stretch",
                alt=f"Aperçu des premières lignes de {uploaded_file.name}",
            )
            for curve_index in range(int(curve_count)):
                st.markdown(f"**Courbe {curve_index + 1}**")
                curve_key = f"{index}_{curve_index}"
                source_columns = st.columns(4)
                strain_source = source_columns[0].selectbox(
                    "Grandeur en X",
                    ["Déformation", "Déplacement"],
                    key=f"strain_source_{curve_key}",
                )
                strain_column = source_columns[1].selectbox(
                    "Colonne en X",
                    options,
                    key=f"strain_col_{curve_key}",
                )
                stress_source = source_columns[2].selectbox(
                    "Grandeur en Y",
                    ["Contrainte déjà calculée", "Force"],
                    key=f"stress_source_{curve_key}",
                )
                stress_column = source_columns[3].selectbox(
                    "Colonne en Y",
                    options,
                    index=min(1, len(options) - 1),
                    key=f"stress_col_{curve_key}",
                )

                curve_details = st.columns(4)
                if strain_source == "Déplacement":
                    gauge_length = curve_details[0].number_input(
                        "Longueur initiale L₀ (mm)",
                        min_value=0.000001,
                        value=50.0,
                        step=1.0,
                        key=f"gauge_length_{curve_key}",
                    )
                    strain_unit = None
                else:
                    gauge_length = None
                    strain_unit = curve_details[0].selectbox(
                        "Unité de déformation",
                        ["Sans dimension (mm/mm)", "Pourcentage (%)"],
                        key=f"strain_unit_{curve_key}",
                    )

                if stress_source == "Force":
                    force_unit = curve_details[1].selectbox(
                        "Unité de force", ["N", "kN"], key=f"force_unit_{curve_key}"
                    )
                    width = curve_details[2].number_input(
                        "Largeur initiale (mm)",
                        min_value=0.000001,
                        value=10.0,
                        step=0.5,
                        key=f"test_width_{curve_key}",
                    )
                    thickness = curve_details[3].number_input(
                        "Épaisseur initiale (mm)",
                        min_value=0.000001,
                        value=2.0,
                        step=0.1,
                        key=f"test_thickness_{curve_key}",
                    )
                    curve_details[3].caption(
                        f"Section : **{width * thickness:.4g} mm²**"
                    )
                    stress_unit = None
                else:
                    force_unit = None
                    width = None
                    thickness = None
                    stress_unit = curve_details[1].selectbox(
                        "Unité de contrainte",
                        ["MPa (N/mm²)", "Pa", "kPa", "GPa"],
                        key=f"stress_unit_{curve_key}",
                    )

                appearance = st.columns(2)
                label = appearance[0].text_input(
                    "Nom de la courbe",
                    f"{Path(uploaded_file.name).stem} — {curve_index + 1}",
                    key=f"curve_label_{curve_key}",
                )
                color = appearance[1].color_picker(
                    "Couleur",
                    palette[(index + curve_index) % len(palette)],
                    key=f"curve_color_{curve_key}",
                )
                if strain_column == stress_column:
                    st.warning(
                        f"Choisissez deux colonnes différentes pour la courbe "
                        f"{curve_index + 1} de **{uploaded_file.name}**."
                    )
                    continue
                curve_configs.append(
                    {
                        "data": data,
                        "strain_column": strain_column,
                        "stress_column": stress_column,
                        "strain_source": strain_source,
                        "stress_source": stress_source,
                        "gauge_length": gauge_length,
                        "strain_unit": strain_unit,
                        "force_unit": force_unit,
                        "width": width,
                        "thickness": thickness,
                        "stress_unit": stress_unit,
                        "label": label,
                        "color": color,
                        "filename": uploaded_file.name,
                    }
                )

    for curve_config in curve_configs:
        data = curve_config["data"]
        assert isinstance(data, pd.DataFrame)
        filename = str(curve_config["filename"])
        strain_column = str(curve_config["strain_column"])
        stress_column = str(curve_config["stress_column"])
        strain_source = str(curve_config["strain_source"])
        stress_source = str(curve_config["stress_source"])
        label = str(curve_config["label"])
        color = str(curve_config["color"])
        strain_values = numeric_column(data, strain_column).to_numpy(dtype=float)
        stress_values = numeric_column(data, stress_column).to_numpy(dtype=float)
        if strain_source == "Déplacement":
            gauge_length = float(curve_config["gauge_length"])
            strain_values = strain_values / gauge_length
        elif curve_config["strain_unit"] == "Pourcentage (%)":
            strain_values = strain_values / 100.0

        if stress_source == "Force":
            force_values = stress_values * (
                1000.0 if curve_config["force_unit"] == "kN" else 1.0
            )
            width = float(curve_config["width"])
            thickness = float(curve_config["thickness"])
            stress_values = force_values / (width * thickness)
        else:
            stress_unit = str(curve_config["stress_unit"])
            stress_values = stress_values * stress_factors[stress_unit]

        valid = np.isfinite(strain_values) & np.isfinite(stress_values)
        strain_values = strain_values[valid]
        stress_values = stress_values[valid]
        if len(strain_values) < 2:
            st.warning(f"Pas assez de valeurs numériques valides dans **{filename}**.")
            continue

        figure.add_trace(
            go.Scatter(
                x=strain_values,
                y=stress_values,
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 2},
                marker={"size": 5},
                hovertemplate="ε = %{x:.6g}<br>σ = %{y:.6g} MPa<extra>%{fullData.name}</extra>",
            )
        )

        elastic = (
            (strain_values >= strain_min)
            & (strain_values <= strain_max)
            & np.isfinite(strain_values)
            & np.isfinite(stress_values)
        )
        young: float | None = None
        intercept: float | None = None
        yield_point: tuple[float, float] | None = None
        if int(np.count_nonzero(elastic)) >= 2 and np.ptp(strain_values[elastic]) > 0:
            young, intercept = (float(value) for value in np.polyfit(
                strain_values[elastic], stress_values[elastic], 1
            ))
            elastic_x = np.linspace(strain_min, strain_max, 50)
            figure.add_trace(
                go.Scatter(
                    x=elastic_x,
                    y=young * elastic_x + intercept,
                    mode="lines",
                    name=f"Young {label} ({young:.4g} MPa)",
                    line={"color": color, "dash": "dash", "width": 2},
                    hovertemplate="ε = %{x:.6g}<br>σ = %{y:.6g} MPa<extra>%{fullData.name}</extra>",
                )
            )
            elastic_min_for_zoom.append(float(np.min(strain_values[elastic])))
            elastic_max_for_zoom.append(float(np.max(strain_values[elastic])))
            if show_offset:
                yield_point = calculate_offset_yield(
                    strain_values, stress_values, young, intercept
                )
                direction = (
                    1.0
                    if abs(float(np.max(strain_values))) >= abs(float(np.min(strain_values)))
                    else -1.0
                )
                offset_start = direction * 0.002
                offset_end = (
                    float(np.max(strain_values))
                    if direction > 0
                    else float(np.min(strain_values))
                )
                if direction * (offset_end - offset_start) >= 0:
                    offset_x = np.linspace(offset_start, offset_end, 100)
                    figure.add_trace(
                        go.Scatter(
                            x=offset_x,
                            y=young * (offset_x - offset_start) + intercept,
                            mode="lines",
                            name=f"Décalage 0,2 % — {label}",
                            line={"color": color, "dash": "dot", "width": 1.5},
                        )
                    )
                if yield_point is not None:
                    figure.add_trace(
                        go.Scatter(
                            x=[yield_point[0]],
                            y=[yield_point[1]],
                            mode="markers",
                            name=f"Rp0,2 {label}",
                            marker={"color": color, "size": 10, "symbol": "x"},
                            hovertemplate="ε = %{x:.6g}<br>Rp0,2 = %{y:.6g} MPa<extra></extra>",
                        )
                    )

        metrics.append(
            {
                "Courbe": label,
                "Rm traction (MPa)": (
                    float(np.max(stress_values[stress_values >= 0]))
                    if np.any(stress_values >= 0)
                    else None
                ),
                "Rc compression (MPa)": (
                    float(abs(np.min(stress_values[stress_values < 0])))
                    if np.any(stress_values < 0)
                    else None
                ),
                "Module d’Young (MPa)": young,
                "Rp0,2 (MPa)": yield_point[1] if yield_point is not None else None,
                "Points valides": len(strain_values),
            }
        )
        if young is None:
            st.warning(
                f"Zone élastique insuffisante pour **{label}** : vérifiez ses bornes "
                "ou les unités de déformation."
            )
        elif show_offset and yield_point is None:
            st.info(f"Aucune intersection Rp0,2 détectée pour **{label}** dans les données.")

    if metrics:
        figure.update_layout(
            title=curve_title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            hovermode="x unified",
            legend_title="Courbes et analyses",
            margin={"l": 30, "r": 25, "t": 70, "b": 40},
        )
        if zoom_elastic and elastic_min_for_zoom and not manual_limits:
            figure.update_xaxes(range=[min(elastic_min_for_zoom), max(elastic_max_for_zoom)])
        elif manual_limits and not zoom_elastic:
            if x_max > x_min:
                figure.update_xaxes(range=[x_min, x_max])
            else:
                st.warning("X max doit être supérieur à X min.")
            if y_max > y_min:
                figure.update_yaxes(range=[y_min, y_max])
            else:
                st.warning("Y max doit être supérieur à Y min.")
        show_plot_downloads(figure, "courbes_traction_compression", "tensile_chart")
        st.subheader("Résultats")
        st.dataframe(
            pd.DataFrame(metrics),
            width="stretch",
            hide_index=True,
            alt="Résultats des résistances, modules d’Young et limites d’élasticité des essais.",
        )
    else:
        st.info("Aucune courbe exploitable. Vérifiez les fichiers et les colonnes sélectionnées.")


def show_composite_analysis() -> None:
    st.header("Matériaux composites")
    st.markdown(
        "Estimation des propriétés d’un composite unidirectionnel à partir des propriétés "
        "de la fibre et de la matrice, puis évolution des modules avec l’angle d’orientation."
    )

    st.subheader("Propriétés des constituants")
    columns = st.columns(5)
    fiber_modulus = columns[0].number_input(
        "Module fibre Ef (GPa)", min_value=0.000001, value=380.0, step=1.0, key="fiber_modulus"
    )
    fiber_poisson = columns[1].number_input(
        "Poisson fibre νf", min_value=-0.99, max_value=0.49, value=0.33, step=0.01,
        key="fiber_poisson",
    )
    matrix_modulus = columns[2].number_input(
        "Module matrice Em (GPa)", min_value=0.000001, value=3.5, step=0.1,
        key="matrix_modulus",
    )
    matrix_poisson = columns[3].number_input(
        "Poisson matrice νm", min_value=-0.99, max_value=0.49, value=0.4, step=0.01,
        key="matrix_poisson",
    )
    fiber_fraction = columns[4].number_input(
        "Fraction volumique Vf", min_value=0.0, max_value=1.0, value=0.6, step=0.01,
        key="fiber_fraction",
        help="Valeur comprise entre 0 et 1 (ex. 60 % = 0,60).",
    )

    matrix_fraction = 1.0 - fiber_fraction
    longitudinal = fiber_fraction * fiber_modulus + matrix_fraction * matrix_modulus
    poisson_lt = fiber_fraction * fiber_poisson + matrix_fraction * matrix_poisson
    fiber_shear = fiber_modulus / (2.0 * (1.0 + fiber_poisson))
    matrix_shear = matrix_modulus / (2.0 * (1.0 + matrix_poisson))
    shear_lt = 1.0 / (fiber_fraction / fiber_shear + matrix_fraction / matrix_shear)
    transverse = 1.0 / (
        fiber_fraction / fiber_modulus + matrix_fraction / matrix_modulus
    )

    result_cols = st.columns(5)
    result_cols[0].metric("El longitudinal", f"{longitudinal:.4g} GPa")
    result_cols[1].metric("Et transverse", f"{transverse:.4g} GPa")
    result_cols[2].metric("Glt cisaillement", f"{shear_lt:.4g} GPa")
    result_cols[3].metric("νlt", f"{poisson_lt:.4g}")
    result_cols[4].metric("Gf / Gm", f"{fiber_shear:.3g} / {matrix_shear:.3g} GPa")

    angle = np.linspace(0.0, 90.0, 181)
    radians = np.radians(angle)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    cos2_sin2 = (cosine * sine) ** 2
    compliance_x = (
        cosine**4 / longitudinal
        + sine**4 / transverse
        + cos2_sin2 * (1.0 / shear_lt - 2.0 * poisson_lt / longitudinal)
    )
    compliance_y = (
        sine**4 / longitudinal
        + cosine**4 / transverse
        + cos2_sin2 * (1.0 / shear_lt - 2.0 * poisson_lt / longitudinal)
    )
    compliance_xy = (
        4.0 * cos2_sin2 * (1.0 / longitudinal + 1.0 / transverse
                          + 2.0 * poisson_lt / longitudinal)
        + (cosine**2 - sine**2) ** 2 / shear_lt
    )
    ex = 1.0 / compliance_x
    ey = 1.0 / compliance_y
    gxy = 1.0 / compliance_xy

    display = st.radio(
        "Grandeurs affichées",
        ["Modules directionnels (GPa)", "Modules normalisés"],
        horizontal=True,
        key="composite_display",
    )
    if display == "Modules normalisés":
        curves = [(ex / longitudinal, "Ex / El", "#176B87"),
                  (ey / longitudinal, "Ey / El", "#E76F51"),
                  (gxy / shear_lt, "Gxy / Glt", "#2A9D8F")]
        y_title = "Module normalisé"
    else:
        curves = [(ex, "Ex", "#176B87"), (ey, "Ey", "#E76F51"), (gxy, "Gxy", "#2A9D8F")]
        y_title = "Module (GPa)"

    composite_figure = go.Figure()
    for values, name, color in curves:
        composite_figure.add_trace(
            go.Scatter(
                x=angle,
                y=values,
                mode="lines",
                name=name,
                line={"color": color, "width": 3},
                hovertemplate="θ = %{x:.1f}°<br>%{y:.5g}<extra>%{fullData.name}</extra>",
            )
        )
    composite_title = st.text_input(
        "Titre du graphique", "Modules élastiques selon l'orientation des fibres",
        key="composite_title",
    )
    composite_x = st.text_input("Titre de l'axe X", "Angle θ (degrés)", key="composite_xlabel")
    composite_y = st.text_input("Titre de l'axe Y", y_title, key="composite_ylabel")
    composite_figure.update_layout(
        title=composite_title,
        xaxis_title=composite_x,
        yaxis_title=composite_y,
        template="plotly_white",
        hovermode="x unified",
        legend_title="Direction",
    )
    show_plot_downloads(composite_figure, "modules_composite", "composite_chart")
    st.caption(
        f"À 0° : Ex = {ex[0]:.4g} GPa ; à 90° : Ey = {ey[-1]:.4g} GPa ; "
        f"à 45° : Gxy = {gxy[len(gxy) // 2]:.4g} GPa."
    )

def show_material_strength() -> None:
    st.header("Résistance des matériaux")
    st.markdown(
        "Calculs de résistance des poutres et colonnes selon les sections et les matériaux."
    )
    st.subheader("Calculs de flexion et de flambement")
    st.info(
        "Cette section est en cours de développement. Les calculs seront bientôt disponibles "
        "pour les sections rectangulaires, circulaires et en I, avec différents matériaux."
    )
def show_help() -> None:
    st.header("Notes d'utilisation")
    st.markdown(
        """
        ### 1. Calculs simples et essais de traction / compression
        La calculatrice rapide utilise une section rectangulaire, `A₀ = largeur x épaisseur`.
        Pour un fichier d'essai, chargez un ou plusieurs CSV, puis configurez chaque courbe
        séparément. Un même fichier peut fournir plusieurs courbes en choisissant des paires
        de colonnes différentes.

        - **Contrainte déjà calculée** : choisissez son unité ; elle est convertie en MPa sans
          nouvelle division par la section.
        - **Force** : saisissez la largeur et l'épaisseur initiales en mm. Avec une force en N,
          `σ nominale (MPa) = F (N) / A₀ (mm²)` ; sélectionnez kN si nécessaire.
        - **Déplacement** : saisissez la longueur initiale de la courbe ; `ε = ΔL / L₀`.
        - **Déformation** : choisissez si les valeurs sont sans dimension ou en pourcentage.
        - **Module d'Young** : pente d'une régression linéaire dans l'intervalle de déformation
          choisi. Ajustez les bornes pour ne garder que la partie élastique.
        - **Rm / Rc** : maxima en traction et magnitude de la contrainte en compression,
          respectivement. La compression demeure signée sur les courbes.
        - **Rp0,2** : première intersection détectée avec la droite parallèle décalée de 0,002
          dans la direction de chargement. Cette valeur est indicative et dépend de la qualité
          de la mesure.

        ### 2. Matériaux composites
        Les calculs reprennent les relations utilisées dans `Analyse.py` : loi des mélanges
        longitudinale, loi inverse transverse, moyenne de Poisson et loi inverse pour le
        cisaillement. Les courbes directionnelles selon θ proviennent des compliances
        transformées du modèle unidirectionnel orthotrope.

        Ces relations donnent une **estimation de pli unidirectionnel** ; elles ne remplacent
        pas un calcul complet de stratifié (séquence d'empilement, couplages ABD, endommagement).
        Tous les modules des constituants doivent être saisis en GPa, et `Vf` entre 0 et 1.

        ### 3. Courbes et export
        Les graphiques sont interactifs : faites glisser la souris pour zoomer sur une zone,
        utilisez la molette pour zoomer (si activée), puis double-cliquez pour revenir à la vue
        initiale. La barre d'outils du graphique offre l'export PNG ; le bouton sous le graphique
        télécharge une version HTML interactive.

        Les CSV peuvent être séparés par `;`, `,`, une tabulation, ou détectés automatiquement.
        Les valeurs non numériques sont ignorées dans les calculs. Vérifiez les unités, le
        séparateur de champs, le séparateur décimal et les colonnes si une courbe ou une pente
        n'apparaît pas.
        """
    )


st.markdown(
    """
    <div class="hero">
        <h1>MecaLab</h1>
        <p>Calculs mécaniques et visualisation rapide de vos essais matériaux</p>
    </div>
    """,
    unsafe_allow_html=True,
)

calculation_tab, composite_tab, material_tab, help_tab = st.tabs(
    ["Essais & calculs simples", "Matériaux composites","Résistance des matériaux", "Notes d'utilisation"]
)
with calculation_tab:
    show_tensile_analysis()
with composite_tab:
    show_composite_analysis()
with material_tab:
    show_material_strength()
with help_tab:
    show_help()
