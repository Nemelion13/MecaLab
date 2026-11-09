"""Analyse linéaire par éléments finis de treillis plans articulés."""

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, sin

import numpy as np
import plotly.graph_objects as go
import streamlit as st


BUNDARIES_DISPLACEMENTS = {
    "Encastrement": (True, True),
    "Support_x": (True, False),
    "Support_y": (False, True),
    "NA": (False, False),
}
SUPPORT_LABELS = {
    "Libre": "NA",
    "Bloqué en x et y": "Encastrement",
    "Bloqué en x": "Support_x",
    "Bloqué en y": "Support_y",
}
RADIUS = 0.3


@dataclass(frozen=True)
class Knuckle:
    id: int
    x: float
    y: float
    bundary_displacement: str
    strength_x: float
    strength_y: float


class Bar:
    def __init__(
        self,
        id: int,
        knuckle1: Knuckle,
        knuckle2: Knuckle,
        young_modulus: float,
        area: float,
    ) -> None:
        self.id = id
        self.knuckle1 = knuckle1
        self.knuckle2 = knuckle2
        self.young_modulus = float(young_modulus)
        self.area = float(area)
        self.length = hypot(knuckle2.x - knuckle1.x, knuckle2.y - knuckle1.y)
        if not np.isfinite(self.length) or self.length <= 0:
            raise ValueError(f"La barre {id} doit relier deux nœuds distincts.")
        if not np.isfinite(self.young_modulus) or self.young_modulus <= 0:
            raise ValueError("Le module d'Young doit être strictement positif.")
        if not np.isfinite(self.area) or self.area <= 0:
            raise ValueError("La section des barres doit être strictement positive.")

        dx = knuckle2.x - knuckle1.x
        dy = knuckle2.y - knuckle1.y
        self.orientation = degrees(atan2(dy, dx))
        self.stiffness_matrix = self._calculate_stiffness_matrix()

    def update_material_properties(self, young_modulus: float, area: float) -> None:
        if not np.isfinite(young_modulus) or young_modulus <= 0:
            raise ValueError("Le module d'Young doit être strictement positif.")
        if not np.isfinite(area) or area <= 0:
            raise ValueError("La section des barres doit être strictement positive.")
        self.young_modulus = float(young_modulus)
        self.area = float(area)
        self.stiffness_matrix = self._calculate_stiffness_matrix()

    def _calculate_stiffness_matrix(self) -> np.ndarray:
        angle = atan2(
            self.knuckle2.y - self.knuckle1.y,
            self.knuckle2.x - self.knuckle1.x,
        )
        c = cos(angle)
        s = sin(angle)
        direction = np.array([-c, -s, c, s], dtype=float)
        area_m2 = self.area * 1e-6
        length_m = self.length * 1e-3
        return (self.young_modulus * area_m2 / length_m) * np.outer(
            direction, direction
        )


@dataclass(frozen=True)
class TrussResult:
    displacements: np.ndarray
    reactions: np.ndarray
    axial_forces: dict[int, float]
    stresses: dict[int, float]


class Truss:
    def __init__(self, knuckles: list[Knuckle], bars: list[Bar]) -> None:
        self.knuckles = list(knuckles)
        self.bars = list(bars)
        self._node_indices = {knuckle.id: index for index, knuckle in enumerate(self.knuckles)}
        if len(self._node_indices) != len(self.knuckles):
            raise ValueError("Chaque nœud doit avoir un identifiant unique.")
        if len({bar.id for bar in self.bars}) != len(self.bars):
            raise ValueError("Chaque barre doit avoir un identifiant unique.")
        for knuckle in self.knuckles:
            if knuckle.bundary_displacement not in BUNDARIES_DISPLACEMENTS:
                raise ValueError(f"Type d'appui inconnu au nœud {knuckle.id}.")
            values = (knuckle.x, knuckle.y, knuckle.strength_x, knuckle.strength_y)
            if not all(np.isfinite(value) for value in values):
                raise ValueError(f"Coordonnées ou forces invalides au nœud {knuckle.id}.")
        for bar in self.bars:
            if (
                bar.knuckle1.id not in self._node_indices
                or bar.knuckle2.id not in self._node_indices
            ):
                raise ValueError(f"La barre {bar.id} référence un nœud absent du treillis.")

        self.assembly_matrix = self.assembly_final_matrix()

    def calculate_static_coefficient(self) -> int:
        """Retourne m + r - 2j, indicateur de détermination statique."""
        reaction_count = sum(
            sum(BUNDARIES_DISPLACEMENTS[knuckle.bundary_displacement])
            for knuckle in self.knuckles
        )
        return len(self.bars) + reaction_count - 2 * len(self.knuckles)

    def assembly_final_matrix(self) -> np.ndarray:
        """Assemble la matrice globale de rigidité (deux ddl par nœud)."""
        degree_count = 2 * len(self.knuckles)
        matrix = np.zeros((degree_count, degree_count), dtype=float)
        for bar in self.bars:
            first = self._node_indices[bar.knuckle1.id]
            second = self._node_indices[bar.knuckle2.id]
            dofs = [2 * first, 2 * first + 1, 2 * second, 2 * second + 1]
            matrix[np.ix_(dofs, dofs)] += bar.stiffness_matrix
        return matrix

    def solve(self) -> TrussResult:
        """Résout les déplacements et calcule réactions et efforts axiaux."""
        if not self.knuckles or not self.bars:
            raise ValueError("Ajoutez au moins un nœud et une barre avant le calcul.")

        force_vector = np.zeros(2 * len(self.knuckles), dtype=float)
        fixed_dofs: list[int] = []
        for index, knuckle in enumerate(self.knuckles):
            force_vector[2 * index] = knuckle.strength_x
            force_vector[2 * index + 1] = knuckle.strength_y
            fixed_x, fixed_y = BUNDARIES_DISPLACEMENTS[
                knuckle.bundary_displacement
            ]
            if fixed_x:
                fixed_dofs.append(2 * index)
            if fixed_y:
                fixed_dofs.append(2 * index + 1)

        free_dofs = np.setdiff1d(np.arange(len(force_vector)), fixed_dofs)
        displacement_vector = np.zeros_like(force_vector)
        if free_dofs.size:
            reduced_matrix = self.assembly_matrix[np.ix_(free_dofs, free_dofs)]
            if np.linalg.matrix_rank(reduced_matrix) < len(free_dofs):
                raise ValueError(
                    "Le treillis est instable ou insuffisamment appuyé "
                    "(mécanisme détecté). Vérifiez la géométrie et les appuis."
                )
            try:
                displacement_vector[free_dofs] = np.linalg.solve(
                    reduced_matrix, force_vector[free_dofs]
                )
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "Le système de rigidité ne peut pas être résolu."
                ) from error

        reaction_vector = self.assembly_matrix @ displacement_vector - force_vector
        reaction_vector[free_dofs] = 0.0
        axial_forces: dict[int, float] = {}
        stresses: dict[int, float] = {}
        for bar in self.bars:
            first = self._node_indices[bar.knuckle1.id]
            second = self._node_indices[bar.knuckle2.id]
            dofs = [2 * first, 2 * first + 1, 2 * second, 2 * second + 1]
            angle = atan2(
                bar.knuckle2.y - bar.knuckle1.y,
                bar.knuckle2.x - bar.knuckle1.x,
            )
            direction = np.array(
                [-cos(angle), -sin(angle), cos(angle), sin(angle)]
            )
            extension_m = float(direction @ displacement_vector[dofs])
            force_n = (
                bar.young_modulus
                * bar.area
                * 1e-6
                / (bar.length * 1e-3)
                * extension_m
            )
            axial_forces[bar.id] = force_n
            stresses[bar.id] = force_n / (bar.area * 1e-6)

        return TrussResult(
            displacements=displacement_vector.reshape((-1, 2)) * 1e3,
            reactions=reaction_vector.reshape((-1, 2)),
            axial_forces=axial_forces,
            stresses=stresses,
        )


def _material_input(
    young_modulus: float = 200e9, area: float = 100.0
) -> dict[str, float]:
    """Saisit les propriétés communes des barres."""
    columns = st.columns(2)
    material_data = {
        "young_modulus": columns[0].number_input(
            "Module d'Young (Pa)",
            value=young_modulus,
            min_value=1.0,
            format="%.3e",
            key="truss_young_modulus",
        ),
        "area": columns[1].number_input(
            "Section (mm²)",
            value=area,
            min_value=1.0,
            key="truss_area",
        ),
    }
    return material_data


def _add_node_form() -> None:
    with st.form("truss_add_node_form", clear_on_submit=True):
        st.caption("Nouveau nœud")
        columns = st.columns(5)
        x = columns[0].number_input("x (mm)", key="truss_node_x")
        y = columns[1].number_input("y (mm)", key="truss_node_y")
        force_x = columns[2].number_input("Fx (N)", key="truss_node_fx")
        force_y = columns[3].number_input("Fy (N)", key="truss_node_fy")
        support_label = columns[4].selectbox(
            "Appui",
            options=list(SUPPORT_LABELS),
            key="truss_node_support",
        )
        submitted = st.form_submit_button("Ajouter le nœud")

    if submitted:
        knuckles = st.session_state["truss_knuckles"]
        node_id = max((knuckle.id for knuckle in knuckles), default=0) + 1
        knuckles.append(
            Knuckle(
                node_id,
                x,
                y,
                SUPPORT_LABELS[support_label],
                force_x,
                force_y,
            )
        )
        st.session_state["truss_calculated"] = False


def _add_bar_form(material_data: dict[str, float]) -> None:
    knuckles = st.session_state["truss_knuckles"]
    if len(knuckles) < 2:
        st.info("Ajoutez au moins deux nœuds pour définir une barre.")
        return

    with st.form("truss_add_bar_form", clear_on_submit=True):
        st.caption("Nouvelle barre")
        node_labels = {
            f"Nœud {knuckle.id} ({knuckle.x:g} ; {knuckle.y:g} mm)": knuckle.id
            for knuckle in knuckles
        }
        columns = st.columns(2)
        first_label = columns[0].selectbox(
            "Nœud de départ", options=list(node_labels), key="truss_bar_start"
        )
        second_label = columns[1].selectbox(
            "Nœud d'arrivée", options=list(node_labels), key="truss_bar_end"
        )
        submitted = st.form_submit_button("Ajouter la barre")

    if submitted:
        if node_labels[first_label] == node_labels[second_label]:
            st.error("Une barre doit relier deux nœuds distincts.")
            return
        by_id = {knuckle.id: knuckle for knuckle in knuckles}
        bars = st.session_state["truss_bars"]
        bar_id = max((bar.id for bar in bars), default=0) + 1
        try:
            bars.append(
                Bar(
                    bar_id,
                    by_id[node_labels[first_label]],
                    by_id[node_labels[second_label]],
                    material_data["young_modulus"],
                    material_data["area"],
                )
            )
            st.session_state["truss_calculated"] = False
        except ValueError as error:
            st.error(str(error))


def _truss_figure(truss: Truss, result: TrussResult, scale: float) -> go.Figure:
    figure = go.Figure()
    initial_nodes = truss.knuckles
    max_force = max(
        (
            np.hypot(node.strength_x, node.strength_y)
            for node in initial_nodes
        ),
        default=0.0,
    )
    max_span = max(
        (
            np.ptp([node.x for node in initial_nodes]),
            np.ptp([node.y for node in initial_nodes]),
        ),
        default=0.0,
    )
    arrow_length = max_span * 0.18 if max_span > 0 else 1.0
    if max_force > 0:
        for node in initial_nodes:
            force_magnitude = float(np.hypot(node.strength_x, node.strength_y))
            if force_magnitude == 0:
                continue
            arrow_scale = arrow_length * force_magnitude / max_force
            figure.add_annotation(
                x=node.x + arrow_scale * node.strength_x / force_magnitude,
                y=node.y + arrow_scale * node.strength_y / force_magnitude,
                ax=node.x,
                ay=node.y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text=f"F ({node.strength_x:g}, {node.strength_y:g}) N",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#d62728",
                font={"color": "#d62728"},
            )

    for bar in truss.bars:
        first_index = truss._node_indices[bar.knuckle1.id]
        second_index = truss._node_indices[bar.knuckle2.id]
        first = bar.knuckle1
        second = bar.knuckle2
        figure.add_trace(
            go.Scatter(
                x=[first.x, second.x],
                y=[first.y, second.y],
                mode="lines",
                line={"color": "#8a8f98", "width": 3, "dash": "dash"},
                name="Géométrie initiale",
                legendgroup="initial",
                showlegend=bar is truss.bars[0],
                hovertemplate="Barre %{fullData.name}<extra></extra>",
            )
        )
        displaced_first = (
            first.x + scale * result.displacements[first_index, 0],
            first.y + scale * result.displacements[first_index, 1],
        )
        displaced_second = (
            second.x + scale * result.displacements[second_index, 0],
            second.y + scale * result.displacements[second_index, 1],
        )
        figure.add_trace(
            go.Scatter(
                x=[displaced_first[0], displaced_second[0]],
                y=[displaced_first[1], displaced_second[1]],
                mode="lines+markers",
                line={"color": "#176b87", "width": 4},
                marker={"size": 8},
                name="Géométrie déformée",
                legendgroup="deformed",
                showlegend=bar is truss.bars[0],
                hovertemplate=f"Barre {bar.id}<extra></extra>",
            )
        )

    figure.update_layout(
        title="Géométrie initiale et déformée",
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        template="plotly_white",
        legend={"orientation": "h"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        hovermode="closest",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def show_truss_analysis() -> None:
    st.header("Treillis plan")
    st.markdown(
        "Modélisation de barres articulées soumises uniquement à un effort axial. "
        "Chaque nœud possède deux degrés de liberté (translations x et y)."
    )
    st.caption(
        "Les coordonnées sont en mm, les forces en N, le module d'Young en Pa "
        "et les sections en mm². Les rotations et le poids propre ne sont pas modélisés."
    )

    st.session_state.setdefault("truss_knuckles", [])
    st.session_state.setdefault("truss_bars", [])
    st.session_state.setdefault("truss_calculated", False)
    material_data = _material_input()
    material_signature = (
        material_data["young_modulus"],
        material_data["area"],
    )
    previous_material = st.session_state.get("truss_material_signature")
    if previous_material is not None and previous_material != material_signature:
        st.session_state["truss_calculated"] = False
    st.session_state["truss_material_signature"] = material_signature
    for bar in st.session_state["truss_bars"]:
        bar.update_material_properties(
            material_data["young_modulus"], material_data["area"]
        )

    st.subheader("1. Nœuds et appuis")
    st.caption(
        "« Bloqué en x et y » fixe les deux translations du nœud ; "
        "les deux autres choix ne bloquent que la direction indiquée."
    )
    _add_node_form()
    if st.session_state["truss_knuckles"]:
        node_rows = [
            {
                "Nœud": node.id,
                "x (mm)": node.x,
                "y (mm)": node.y,
                "Appui": next(
                    label
                    for label, code in SUPPORT_LABELS.items()
                    if code == node.bundary_displacement
                ),
                "Fx (N)": node.strength_x,
                "Fy (N)": node.strength_y,
            }
            for node in st.session_state["truss_knuckles"]
        ]
        st.dataframe(
            node_rows,
            hide_index=True,
            width="stretch",
            alt="Nœuds du treillis, coordonnées, appuis et charges appliquées",
        )

    st.subheader("2. Barres")
    _add_bar_form(material_data)
    if st.session_state["truss_bars"]:
        st.dataframe(
            [
                {
                    "Barre": bar.id,
                    "Nœuds": f"{bar.knuckle1.id} - {bar.knuckle2.id}",
                    "Longueur (mm)": bar.length,
                    "E (Pa)": bar.young_modulus,
                    "Section (mm²)": bar.area,
                }
                for bar in st.session_state["truss_bars"]
            ],
            hide_index=True,
            width="stretch",
            alt="Barres du treillis et propriétés géométriques et matérielles",
        )

    if st.button("Effacer le modèle"):
        st.session_state["truss_knuckles"] = []
        st.session_state["truss_bars"] = []
        st.session_state["truss_calculated"] = False
        st.rerun()

    if not st.session_state["truss_bars"]:
        return

    scale = st.slider(
        "Facteur d'amplification de la déformée",
        min_value=1,
        max_value=1000,
        value=50,
    )
    if st.button("Calculer le treillis", type="primary"):
        st.session_state["truss_calculated"] = True
    if not st.session_state["truss_calculated"]:
        return

    try:
        truss = Truss(
            st.session_state["truss_knuckles"],
            st.session_state["truss_bars"],
        )
        result = truss.solve()
    except ValueError as error:
        st.error(str(error))
        return

    coefficient = truss.calculate_static_coefficient()
    if coefficient == 0:
        st.success("Indice m + r - 2j = 0 (condition nécessaire à l'isostaticité).")
    elif coefficient > 0:
        st.info(f"Indice m + r - 2j = {coefficient} (hyperstaticité potentielle).")
    else:
        st.warning(f"Indice m + r - 2j = {coefficient} (structure sous-contrainte).")

    st.plotly_chart(
        _truss_figure(truss, result, scale),
        width="stretch",
        alt="Comparaison du treillis initial et de sa déformée calculée",
    )
    st.caption(
        f"Déformée affichée avec un facteur {scale} ; les déplacements calculés "
        "et les efforts ci-dessous ne sont pas amplifiés."
    )

    displacement_rows = [
        {
            "Nœud": node.id,
            "ux (mm)": result.displacements[index, 0],
            "uy (mm)": result.displacements[index, 1],
            "Réaction Rx (N)": result.reactions[index, 0],
            "Réaction Ry (N)": result.reactions[index, 1],
        }
        for index, node in enumerate(truss.knuckles)
    ]
    st.subheader("Déplacements et réactions")
    st.dataframe(
        displacement_rows,
        hide_index=True,
        width="stretch",
        alt="Déplacements et réactions calculés pour chaque nœud",
    )

    st.subheader("Efforts dans les barres")
    st.caption("Un effort axial positif correspond à la traction ; un effort négatif à la compression.")
    st.dataframe(
        [
            {
                "Barre": bar.id,
                "Effort axial (N)": result.axial_forces[bar.id],
                "Contrainte normale (Pa)": result.stresses[bar.id],
            }
            for bar in truss.bars
        ],
        hide_index=True,
        width="stretch",
        alt="Efforts axiaux et contraintes normales dans chaque barre",
    )
