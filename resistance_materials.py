from __future__ import annotations
from PIL import Image
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class SectionProperties:
    area: float
    inertia: float
    section_modulus: float
    height: float


@dataclass(frozen=True)
class BeamResponse:
    position: np.ndarray
    shear: np.ndarray
    moment: np.ndarray
    deflection: np.ndarray
    reaction_left: float
    reaction_right: float
    max_moment: float
    max_moment_position: float
    max_deflection: float
    max_deflection_position: float


SECTION_TYPES = (
    "Rectangle", 
    "Tube rectangulaire",
    "Rond plein",
    "Tube circulaire",
    "Profilé en I symétrique",
)

BEAM_CASES = (
    "Poutre simplement appuyée — charge ponctuelle au milieu",
    "Poutre simplement appuyée — charge répartie sur toute la portée",
    "Porte-à-faux — charge ponctuelle en extrémité",
    "Porte-à-faux — charge répartie sur toute la longueur",
)

BEAM_CASES_IMAGES = (
    "Images/PSACP.png"
)

def rectangular_section(width: float, height: float) -> SectionProperties:
    if width <= 0 or height <= 0:
        raise ValueError("Les dimensions du rectangle doivent être positives.")
    area = width * height
    inertia = width * height**3 / 12.0
    return SectionProperties(area, inertia, inertia / (height / 2.0), height)

def solid_circular_section(diameter: float) -> SectionProperties:
    if diameter <= 0:
        raise ValueError("Le diamètre doit être positif.")
    area = np.pi * diameter**2 / 4.0
    inertia = np.pi * diameter**4 / 64.0
    return SectionProperties(area, inertia, inertia / (diameter / 2.0), diameter)

def hollow_rectangular_section(
        inner_width: float,
        inner_height: float,
        outer_width: float,
        outer_height: float
) -> SectionProperties:
    if inner_height <= 0 or inner_width <=0 or outer_height < inner_height or outer_width < inner_width:
        raise ValueError("Les dimensions doivent être positives et le rectangle extérieur plus grand que le rectangle intérieur")
    area = (outer_width * outer_height) - (inner_height * inner_width)
    inertia = ((outer_width * outer_height**3) - (inner_width * inner_height**3)) / 12.0
    return SectionProperties(area, inertia,inertia / (outer_height / 2.0), outer_height)
def hollow_circular_section(
    outer_diameter: float,
    inner_diameter: float,
) -> SectionProperties:
    if outer_diameter <= 0 or inner_diameter < 0 or inner_diameter >= outer_diameter:
        raise ValueError("Il faut 0 ≤ diamètre intérieur < diamètre extérieur.")
    area = np.pi * (outer_diameter**2 - inner_diameter**2) / 4.0
    inertia = np.pi * (outer_diameter**4 - inner_diameter**4) / 64.0
    return SectionProperties(
        area,
        inertia,
        inertia / (outer_diameter / 2.0),
        outer_diameter,
    )


def symmetric_i_section(
    width: float,
    height: float,
    flange_thickness: float,
    web_thickness: float,
) -> SectionProperties:
    if min(width, height, flange_thickness, web_thickness) <= 0:
        raise ValueError("Toutes les dimensions du profilé doivent être positives.")
    if 2.0 * flange_thickness >= height or web_thickness >= width:
        raise ValueError(
            "Le profilé doit vérifier 2 × épaisseur de semelle < hauteur "
            "et épaisseur d'âme < largeur."
        )
    area = 2.0 * width * flange_thickness + web_thickness * (
        height - 2.0 * flange_thickness
    )
    inertia = (
        width * height**3
        - (width - web_thickness) * (height - 2.0 * flange_thickness) ** 3
    ) / 12.0
    return SectionProperties(area, inertia, inertia / (height / 2.0), height)


def beam_response(
    #Img: file,
    case: str,
    length: float,
    load: float,
    young_modulus: float,
    inertia: float,
    sample_count: int = 401,
    
) -> BeamResponse:
    if length <= 0 or load <= 0 or young_modulus <= 0 or inertia <= 0:
        raise ValueError("La portée, la charge, E et I doivent être positifs.")
    if case not in BEAM_CASES:
        raise ValueError(f"Cas de poutre non pris en charge : {case}")

    position = np.linspace(0.0, length, sample_count)
    shear = np.zeros_like(position)
    moment = np.zeros_like(position)
    deflection = np.zeros_like(position)
    reaction_left = 0.0
    reaction_right = 0.0

    if case == BEAM_CASES[0]:
        midpoint = length / 2.0
        side_count = max(sample_count // 2, 2)
        position = np.concatenate(
            (
                np.linspace(0.0, midpoint, side_count),
                np.linspace(midpoint, length, side_count),
            )
        )
        shear = np.zeros_like(position)
        moment = np.zeros_like(position)
        deflection = np.zeros_like(position)
        left = position <= midpoint
        right_distance = length - position
        shear[position < midpoint] = load / 2.0
        shear[position >= midpoint] = -load / 2.0
        moment[left] = load * position[left] / 2.0
        moment[~left] = load * right_distance[~left] / 2.0
        deflection[left] = -load * position[left] * (
            3.0 * length**2 - 4.0 * position[left] ** 2
        ) / (48.0 * young_modulus * inertia)
        deflection[~left] = -load * right_distance[~left] * (
            3.0 * length**2 - 4.0 * right_distance[~left] ** 2
        ) / (48.0 * young_modulus * inertia)
        reaction_left = reaction_right = load / 2.0
        max_moment = load * length / 4.0
        max_moment_position = midpoint
        max_deflection = load * length**3 / (48.0 * young_modulus * inertia)
        max_deflection_position = midpoint
        
    elif case == BEAM_CASES[1]:
        reaction_left = reaction_right = load * length / 2.0
        shear = reaction_left - load * position
        moment = reaction_left * position - load * position**2 / 2.0
        deflection = -load * position * (
            length**3 - 2.0 * length * position**2 + position**3
        ) / (24.0 * young_modulus * inertia)
        max_moment = load * length**2 / 8.0
        max_moment_position = length / 2.0
        max_deflection = 5.0 * load * length**4 / (
            384.0 * young_modulus * inertia
        )
        max_deflection_position = length / 2.0
    elif case == BEAM_CASES[2]:
        shear.fill(-load)
        moment = -load * (length - position)
        deflection = -load * position**2 * (
            3.0 * length - position
        ) / (6.0 * young_modulus * inertia)
        reaction_left = load
        max_moment = load * length
        max_moment_position = length
        max_deflection = load * length**3 / (3.0 * young_modulus * inertia)
        max_deflection_position = length
    else:
        shear = -load * (length - position)
        moment = -load * (length - position) ** 2 / 2.0
        deflection = -load * position**2 * (
            6.0 * length**2
            - 4.0 * length * position
            + position**2
        ) / (24.0 * young_modulus * inertia)
        reaction_left = load * length
        max_moment = load * length**2 / 2.0
        max_moment_position = 0.0
        max_deflection = load * length**4 / (
            8.0 * young_modulus * inertia
        )
        max_deflection_position = length

    return BeamResponse(
        position=position,
        shear=shear,
        moment=moment,
        deflection=deflection,
        reaction_left=reaction_left,
        reaction_right=reaction_right,
        max_moment=max_moment,
        max_moment_position=max_moment_position,
        max_deflection=max_deflection,
        max_deflection_position=max_deflection_position,
    )


def _section_inputs(section_type: str) -> tuple[SectionProperties, dict[str, float]]:
    dimensions: dict[str, float] = {}
    if section_type == "Rectangle":
        columns = st.columns(2)
        dimensions["Largeur b (mm)"] = columns[0].number_input(
            "Largeur b (mm)", min_value=0.1, value=20.0, step=1.0,
            key="rdm_rectangle_width",
        )
        dimensions["Hauteur h (mm)"] = columns[1].number_input(
            "Hauteur h (mm)", min_value=0.1, value=40.0, step=1.0,
            key="rdm_rectangle_height",
        )
        properties = rectangular_section(
            dimensions["Largeur b (mm)"], dimensions["Hauteur h (mm)"]
        )

    elif section_type == "Tube rectangulaire":
        columns = st.columns(4)
        dimensions["Largeur intérieure b (mm)"] = columns[0].number_input(
            "Largeur intérieure b (mm)", min_value=0.1, value=20.0, step=1.0,
            key="rdm_tube_rectangle_width_int",
        )
        dimensions["Hauteur intérieure h (mm)"] = columns[1].number_input(
             "Hauteur intérieure h (mm)", min_value=0.1, value=30.0, step=1.0,
            key="rdm_tube_rectangle_height_int",
        )
        
        dimensions["Largeur extérieure b' (mm)"] = columns[2].number_input(
            "largeur extérieure b' (mm)", min_value=0.1, value=30.0, step=1.0,
            key="rdm_tube_rectangle_width_ext",
        )
        dimensions["Hauteur extérieure h' (mm)"] = columns[1].number_input(
            "Hauteur extérieure h' (mm)", min_value=0.1, value=40.0, step=1.0,
            key="rdm_tube_rectangle_height_ext",
            )
        properties = hollow_rectangular_section(
           dimensions["Largeur intérieure b (mm)"],
           dimensions["Hauteur intérieure h (mm)"],
           dimensions["Largeur extérieure b' (mm)"],
           dimensions["Hauteur extérieure h' (mm)"]
        )

    elif section_type == "Rond plein":
        dimensions["Diamètre d (mm)"] = st.number_input(
            "Diamètre d (mm)", min_value=0.1, value=20.0, step=1.0,
            key="rdm_solid_diameter",
        )
        properties = solid_circular_section(dimensions["Diamètre d (mm)"])
    elif section_type == "Tube circulaire":
        columns = st.columns(2)
        dimensions["Diamètre extérieur D (mm)"] = columns[0].number_input(
            "Diamètre extérieur D (mm)", min_value=0.1, value=30.0, step=1.0,
            key="rdm_tube_outer",
        )
        dimensions["Diamètre intérieur d (mm)"] = columns[1].number_input(
            "Diamètre intérieur d (mm)", min_value=0.0, value=20.0, step=1.0,
            key="rdm_tube_inner",
        )
        if dimensions["Diamètre intérieur d (mm)"] >= dimensions["Diamètre extérieur D (mm)"]:
            raise ValueError("Le diamètre intérieur doit être inférieur au diamètre extérieur.")
        properties = hollow_circular_section(
            dimensions["Diamètre extérieur D (mm)"],
            dimensions["Diamètre intérieur d (mm)"],
        )
    else:
        columns = st.columns(4)
        dimensions["Largeur des semelles b (mm)"] = columns[0].number_input(
            "Largeur des semelles b (mm)", min_value=0.1, value=40.0, step=1.0,
            key="rdm_i_width",
        )
        dimensions["Hauteur h (mm)"] = columns[1].number_input(
            "Hauteur totale h (mm)", min_value=0.2, value=80.0, step=1.0,
            key="rdm_i_height",
        )
        dimensions["Épaisseur semelles tf (mm)"] = columns[2].number_input(
            "Épaisseur des semelles tf (mm)", min_value=0.1, value=5.0, step=0.5,
            key="rdm_i_flange",
        )
        dimensions["Épaisseur âme tw (mm)"] = columns[3].number_input(
            "Épaisseur de l'âme tw (mm)", min_value=0.1, value=4.0, step=0.5,
            key="rdm_i_web",
        )
        if 2.0 * dimensions["Épaisseur semelles tf (mm)"] >= dimensions["Hauteur h (mm)"]:
            raise ValueError("Deux semelles doivent tenir dans la hauteur totale du profilé.")
        if dimensions["Épaisseur âme tw (mm)"] >= dimensions["Largeur des semelles b (mm)"]:
            raise ValueError("L'âme doit être moins large que les semelles.")
        properties = symmetric_i_section(
            dimensions["Largeur des semelles b (mm)"],
            dimensions["Hauteur h (mm)"],
            dimensions["Épaisseur semelles tf (mm)"],
            dimensions["Épaisseur âme tw (mm)"],
        )
    return properties, dimensions


def _section_figure(section_type: str, dimensions: dict[str, float]) -> go.Figure:
    figure = go.Figure()
    if section_type in ("Rectangle","Tube rectangulaire"):
        width = ( 
            dimensions["Largeur b (mm)"] 
            if section_type== "Rectangle" 
            else dimensions["Largeur extérieure b' (mm)"]
        )
        height = ( 
                    dimensions["Hauteur h (mm)"] 
                    if section_type== "Rectangle" 
                    else dimensions["Hauteur extérieure h' (mm)"]
                )
        x = [-width / 2, width / 2, width / 2, -width / 2, -width / 2]
        y = [-height / 2, -height / 2, height / 2, height / 2, -height / 2]
        
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": "#176b87", "width": 3},
                fill="toself",
                fillcolor="rgba(23,107,135,0.16)",
                showlegend=False,
                hoverinfo="skip"
            )
        )

        if section_type== "Tube rectangulaire":
            inner_width = dimensions["Largeur intérieure b (mm)"]
            inner_height = dimensions["Hauteur intérieure h (mm)"]
            x_inner = [-inner_width / 2, inner_width / 2, inner_width / 2, -inner_width / 2, -inner_width / 2]
            y_inner = [-inner_height / 2, -inner_height / 2, inner_height / 2, inner_height / 2, -inner_height / 2]
            figure.add_trace(
                go.Scatter(
                    x=x_inner,
                    y=y_inner,
                    mode="lines",
                    line={"color": "#176b87", "width": 2},
                    fill="toself",
                    fillcolor="white", # non visible
                    showlegend=False,
                    hoverinfo="skip"
                )
            )
                        
    elif section_type in ("Rond plein", "Tube circulaire"):
        diameter = (
            dimensions["Diamètre d (mm)"]
            if section_type == "Rond plein"
            else dimensions["Diamètre extérieur D (mm)"]
        )
        angles = np.linspace(0.0, 2.0 * np.pi, 160)
        figure.add_trace(
            go.Scatter(
                x=diameter / 2.0 * np.cos(angles),
                y=diameter / 2.0 * np.sin(angles),
                mode="lines",
                line={"color": "#176b87", "width": 3},
                fill="toself",
                fillcolor="rgba(23,107,135,0.16)",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        if section_type == "Tube circulaire":
            inner = dimensions["Diamètre intérieur d (mm)"]
            figure.add_trace(
                go.Scatter(
                    x=inner / 2.0 * np.cos(angles),
                    y=inner / 2.0 * np.sin(angles),
                    mode="lines",
                    line={"color": "#176b87", "width": 2},
                    fill="toself",
                    fillcolor="white",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        figure.update_layout(
            title=f"Section — {section_type}",
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            template="plotly_white",
            showlegend=False,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
        )
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
        return figure
    else:
        width = dimensions["Largeur des semelles b (mm)"]
        height = dimensions["Hauteur h (mm)"]
        flange = dimensions["Épaisseur semelles tf (mm)"]
        web = dimensions["Épaisseur âme tw (mm)"]
        x = [
            -width / 2, width / 2, width / 2, web / 2, web / 2,
            width / 2, width / 2, -width / 2, -width / 2, -web / 2,
            -web / 2, -width / 2, -width / 2,
        ]
        y = [
            height / 2, height / 2, height / 2 - flange,
            height / 2 - flange, -height / 2 + flange,
            -height / 2 + flange, -height / 2, -height / 2,
            -height / 2 + flange, -height / 2 + flange,
            height / 2 - flange, height / 2 - flange, height / 2,
        ]

        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": "#176b87", "width": 3},
                fill="toself",
                fillcolor="rgba(23,107,135,0.16)",
                showlegend=False,
                hoverinfo="skip",
            )
    )
    figure.update_layout(
        title=f"Section — {section_type}",
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        template="plotly_white",
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def _show_section_characteristics(
    section_type: str,
    properties: SectionProperties,
    dimensions: dict[str, float],
) -> None:
    st.subheader("Caractéristiques de la section")
    left, right = st.columns([1, 1.5])
    with left:
        st.metric("Aire A", f"{properties.area:.5g} mm²")
        st.metric("Moment quadratique Iy", f"{properties.inertia:.5g} mm⁴")
        st.metric("Module de section Wel", f"{properties.section_modulus:.5g} mm³")
        st.caption(
            "Iy est calculé autour de l'axe centroidal horizontal, pour une flexion "
            "dans le plan vertical représenté."
        )
    with right:
        st.plotly_chart(
            _section_figure(section_type, dimensions),
            width="stretch",
            alt=f"Schéma de la section {section_type} utilisé pour les propriétés mécaniques.",
        )


def _show_beam_flexure(
    properties: SectionProperties,
    young_modulus_mpa: float,
    yield_strength: float | None,
    target_factor: float,
) -> None:
    st.subheader("Flexion d'une poutre — cas simples")
    controls = st.columns(2)
    case = controls[0].selectbox("Conditions d'appui et de charge", BEAM_CASES, key="beam_case")
    length = controls[1].number_input(
        "Portée L (mm)", min_value=1.0, value=1000.0, step=50.0, key="beam_length"
    )
    load_label = "Force ponctuelle P (N)" if "ponctuelle" in case else "Charge répartie q (N/mm)"
    load = st.number_input(
        load_label,
        min_value=0.000001,
        value=1000.0 if "ponctuelle" in case else 1.0,
        step=100.0 if "ponctuelle" in case else 0.1,
        key="beam_load",
    )
    response = beam_response(
        case, length, load, young_modulus_mpa, properties.inertia
    )
    max_stress = response.max_moment / properties.section_modulus
    reactions = st.columns(3)
    reactions[0].metric("Moment fléchissant max. |M|max", f"{response.max_moment:.5g} N·mm")
    reactions[1].metric("Contrainte de flexion max. |σ|max", f"{max_stress:.5g} MPa")
    reactions[2].metric(
        "Flèche max. |δ|max",
        f"{response.max_deflection:.5g} mm",
        help=f"Valeur calculée à x = {response.max_deflection_position:.5g} mm.",
    )
    if yield_strength is not None:
        factor_of_safety = yield_strength / max_stress if max_stress > 0 else float("inf")
        st.metric("Coefficient de sécurité indicatif fy / |σ|max", f"{factor_of_safety:.4g}")
        if factor_of_safety < 1.0:
            st.error("La contrainte de flexion calculée dépasse la limite élastique saisie.")
        elif factor_of_safety < target_factor:
            st.warning(
                f"Le coefficient indicatif est inférieur à la cible saisie "
                f"({target_factor:.3g})."
            )
        else:
            st.success("La contrainte de flexion reste sous la limite élastique saisie.")

    response_columns = st.columns(2)
    response_columns[0].metric("Réaction à gauche / encastrement", f"{response.reaction_left:.5g} N")
    if "appuyée" in case:
        response_columns[1].metric("Réaction à droite", f"{response.reaction_right:.5g} N")
    else:
        response_columns[1].caption("Porte-à-faux : réaction verticale à l'extrémité libre nulle.")

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=("Effort tranchant V (N)", "Moment fléchissant M (N·mm)", "Flèche δ (mm)"),
    )
    figure.add_trace(
        go.Scatter(x=response.position, y=response.shear, mode="lines", name="V",
                   line={"color": "#176b87", "width": 2}),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(x=response.position, y=response.moment, mode="lines", name="M",
                   line={"color": "#e76f51", "width": 2}, fill="tozeroy"),
        row=2, col=1,
    )
    figure.add_trace(
        go.Scatter(x=response.position, y=response.deflection, mode="lines", name="δ",
                   line={"color": "#2a9d8f", "width": 2}),
        row=3, col=1,
    )
    figure.update_xaxes(title_text="Position x (mm)", row=3, col=1)
    figure.update_yaxes(zeroline=True, zerolinecolor="#829ab1")
    figure.update_layout(
        template="plotly_white",
        height=700,
        showlegend=False,
        margin={"l": 40, "r": 25, "t": 70, "b": 40},
    )
    st.plotly_chart(
        figure,
        width="stretch",
        alt="Diagrammes de l'effort tranchant, du moment fléchissant et de la déformée de la poutre.",
    )

    load_description = (
        "charge ponctuelle centrée"
        if case == BEAM_CASES[0]
        else "charge uniformément répartie sur la portée"
    )
    if "Porte-à-faux" in case:
        load_description = (
            "charge ponctuelle en bout libre"
            if case == BEAM_CASES[2]
            else "charge uniformément répartie"
        )
    st.caption(
        f"Modèle : {case}. Charge : {load_description}. "
        "La déformée est négative vers le bas ; les valeurs maximales affichées sont des magnitudes."
    )
    st.info(
        "Hypothèses : poutre prismatique, matériau homogène et isotrope, comportement élastique "
        "linéaire, petites déformations, théorie d'Euler–Bernoulli et chargement statique. "
        "La contrainte est la contrainte normale de flexion σ = M/W ; l'effort tranchant n'est "
        "pas converti en contrainte de cisaillement. Les vérifications de flambement, fatigue, "
        "déversement, assemblages et normes de dimensionnement ne sont pas incluses."
    )


def show_material_strength() -> None:
    st.header("Résistance des matériaux")
    st.markdown(
        "Caractérisez une section et estimez la réponse en flexion d'une poutre pour des "
        "conditions d'appui et de chargement prédéfinies."
    )
    section_column, material_column = st.columns([1, 1])
    try:
        with section_column:
            section_type = st.selectbox(
                "Type de section",
                SECTION_TYPES,
                key="rdm_section_type",
            )
            with st.container(border=True):
                properties, dimensions = _section_inputs(section_type)
    except ValueError as error:
        st.error(str(error))
        return

    with material_column:
        st.markdown("**Matériau**")
        modulus_unit = st.selectbox(
            "Unité du module d'Young pour le matériau",
            ["GPa", "MPa"],
            key="rdm_material_modulus_unit",
        )
        default_modulus = 210.0 if modulus_unit == "GPa" else 210000.0
        modulus = st.number_input(
            f"Module d'Young E ({modulus_unit})",
            min_value=0.000001,
            value=default_modulus,
            key=(
                "rdm_material_modulus_gpa"
                if modulus_unit == "GPa"
                else "rdm_material_modulus_mpa"
            ),
        )
        use_yield_strength = st.checkbox(
            "Comparer à une limite élastique fy",
            value=False,
            key="rdm_use_yield_strength",
        )
        yield_strength = (
            st.number_input(
                "Limite élastique fy (MPa)",
                min_value=0.000001,
                value=250.0,
                step=10.0,
                key="rdm_yield_strength",
            )
            if use_yield_strength
            else None
        )
        target_factor = (
            st.number_input(
                "Coefficient de sécurité cible",
                min_value=1.0,
                value=1.5,
                step=0.1,
                key="rdm_target_factor",
            )
            if use_yield_strength
            else 1.0
        )

    st.divider()
    tool = st.segmented_control(
        "Outil",
        ["Caractéristiques de section", "Flexion de poutre"],
        default="Caractéristiques de section",
        key="rdm_tool",
    )
    if tool == "Flexion de poutre":
        modulus_mpa = modulus * 1000.0 if modulus_unit == "GPa" else modulus
        st.caption(
            f"Section active : A = {properties.area:.5g} mm², "
            f"Iy = {properties.inertia:.5g} mm⁴, "
            f"Wel = {properties.section_modulus:.5g} mm³."
        )
        _show_beam_flexure(properties, modulus_mpa, yield_strength, target_factor)
    else:
        _show_section_characteristics(section_type, properties, dimensions)
