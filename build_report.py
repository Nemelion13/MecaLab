import hashlib
from io import StringIO, BytesIO
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

def build_results_csv(metrics: list[dict[str, object]]) -> bytes:
    """Serialize the analysis summary as a French-locale CSV for spreadsheet apps."""
    csv_buffer = StringIO()
    pd.DataFrame(metrics).to_csv(
        csv_buffer,
        index=False,
        sep=";",
        decimal=",",
        lineterminator="\n",
    )
    return csv_buffer.getvalue().encode("utf-8-sig")


def _report_value(value: object, unit: str = "") -> str:
    if value is None:
        return "Non calculé"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric_value):
        return "Non calculé"
    return f"{numeric_value:.5g}{unit}"


def _style_report_axis(ax: plt.Axes, title: str, x_label: str, y_label: str) -> None:
    if title:
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color="#102a43")
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.grid(True, color="#dce8ef", linewidth=0.7)
    ax.tick_params(labelsize=7)
    ax.set_axisbelow(True)


def _set_report_zoom_y_limits(
    ax: plt.Axes,
    curves: list[dict[str, object]],
    x_limits: tuple[float, float],
) -> None:
    visible_stress = [
        np.asarray(curve["stress"], dtype=float)[
            (np.asarray(curve["strain"], dtype=float) >= x_limits[0])
            & (np.asarray(curve["strain"], dtype=float) <= x_limits[1])
        ]
        for curve in curves
    ]
    visible_stress = [values for values in visible_stress if len(values)]
    ax.set_xlim(*x_limits)
    if not visible_stress:
        ax.text(
            0.5, 0.5, "Aucun point dans la plage de zoom",
            ha="center", va="center", transform=ax.transAxes, fontsize=8,
        )
        return
    all_visible = np.concatenate(visible_stress)
    y_min = float(np.min(all_visible))
    y_max = float(np.max(all_visible))
    margin = max((y_max - y_min) * 0.1, max(abs(y_min), abs(y_max)) * 0.04, 1.0)
    ax.set_ylim(y_min - margin, y_max + margin)


def _report_signature(
    curves: list[dict[str, object]],
    comparison: dict[str, object] | None,
    x_label: str,
    y_label: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{x_label}\0{y_label}".encode("utf-8"))
    for curve in curves:
        for key in ("label", "filename", "configuration", "color"):
            digest.update(str(curve[key]).encode("utf-8"))
            digest.update(b"\0")
        for key in ("strain", "stress"):
            values = np.ascontiguousarray(curve[key])
            digest.update(key.encode("ascii"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(repr(values.shape).encode("ascii"))
            digest.update(values.tobytes())
        for key in (
            "metrics",
            "young",
            "intercept",
            "elastic_range",
            "elastic_limit_stress",
            "yield_point",
        ):
            digest.update(repr(curve[key]).encode("utf-8"))
            digest.update(b"\0")
    if comparison is not None:
        for key in ("strain_min", "strain_max", "metrics"):
            digest.update(repr(comparison[key]).encode("utf-8"))
            digest.update(b"\0")
        for key in ("reference", "compared"):
            curve = comparison[key]
            assert isinstance(curve, dict)
            digest.update(key.encode("ascii"))
            digest.update(b"\0")
            digest.update(str(curve["label"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(curve["filename"]).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _plot_report_curve(
    ax: plt.Axes,
    curve: dict[str, object],
    x_label: str,
    y_label: str,
    x_limits: tuple[float, float] | None = None,
    set_zoom_limits: bool = True,
) -> None:
    # Tracé principal
    strain = np.asarray(curve["strain"], dtype=float)
    stress = np.asarray(curve["stress"], dtype=float)
    color = str(curve["color"])
    label = str(curve["label"])
    ax.plot(strain, stress, color=color, linewidth=1.7, label=label, zorder=2)

    #Module d'Young et droite de régression linéaire
    young = curve["young"]
    intercept = curve["intercept"]
    elastic_range = curve["elastic_range"]
    if young is not None and intercept is not None:
        elastic_min, elastic_max = elastic_range
        fit_min = max(float(np.min(strain)), float(elastic_min))
        fit_max = min(float(np.max(strain)), float(elastic_max))
        if fit_max > fit_min:
            fit_strain = np.linspace(fit_min, fit_max, 80)
            ax.plot(
                fit_strain,
                float(young) * fit_strain + float(intercept),
                color=color,
                linestyle="--",
                linewidth=1.1,
                alpha=0.8,
                label=f"Régression élastique (E = {_report_value(young, ' MPa')})",
            )
            elastic_points = strain[(strain >= elastic_min) & (strain <= elastic_max)]
            if len(elastic_points):
                young_marker_x = float(np.mean(elastic_points))
                young_marker_y = float(young) * young_marker_x + float(intercept)
                ax.scatter(
                    [young_marker_x], [young_marker_y], color=color,
                    marker="D", s=30, zorder=5,
                )
                ax.annotate(
                    f"E = {_report_value(young, ' MPa')}",
                    (young_marker_x, young_marker_y), xytext=(5, 7),
                    textcoords="offset points", fontsize=7, color=color,
                )

    tensile = np.flatnonzero(stress >= 0)
    if len(tensile):
        rm_index = int(tensile[np.argmax(stress[tensile])])
        ax.scatter(
            strain[rm_index], stress[rm_index], color=color, marker="o",
            facecolors="none", s=48, zorder=4,
        )
        ax.annotate("Rm", (strain[rm_index], stress[rm_index]), xytext=(5, 6),
                    textcoords="offset points", fontsize=7, color=color)
    compression = np.flatnonzero(stress < 0)
    if len(compression):
        rc_index = int(compression[np.argmin(stress[compression])])
        ax.scatter(
            strain[rc_index], stress[rc_index], color=color, marker="s",
            facecolors="none", s=44, zorder=4,
        )
        ax.annotate("Rc", (strain[rc_index], stress[rc_index]), xytext=(5, -10),
                    textcoords="offset points", fontsize=7, color=color)

    # Contrainte à la limite élastique maximale de déformation
    elastic_limit_stress = curve["elastic_limit_stress"]
    elastic_max = float(curve["elastic_range"][1])
    if elastic_limit_stress is not None:
        ax.scatter(
            [elastic_max], [float(elastic_limit_stress)], color=color,
            marker="*", s=75, zorder=5, label="Re à la borne élastique",
        )

    # Tracer de la limite conventionelle RP0,2
    yield_point = curve["yield_point"]
    if yield_point is not None:
        yield_strain, yield_stress = yield_point
        ax.scatter(
            [yield_strain], [yield_stress], color=color, marker="x",
            linewidths=1.8, s=45, zorder=5, label="Rp0,2",
        )
        direction = (
            1.0
            if abs(float(np.max(strain))) >= abs(float(np.min(strain)))
            else -1.0
        )
        offset_start = direction * 0.002
        offset_end = (
            yield_point[0] if yield_point is not None else direction * 0.01
        )
        if direction * (offset_end - offset_start) >= 0:
            offset_x = np.linspace(offset_start, offset_end, 100)
        ax.plot(
            offset_x,
            float(young) * (offset_x - offset_start) + float(intercept),
            color=color,
            linestyle=":",
            linewidth=1.1,
            alpha=0.8,
            label=f"Décalage à 0,2% (Rp0,2 = {_report_value(yield_stress, ' MPa')})",
        )

    if x_limits is not None:
        if set_zoom_limits:
            _set_report_zoom_y_limits(ax, [curve], x_limits)
        else:
            ax.set_xlim(*x_limits)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6.5, loc="best", frameon=True, framealpha=0.9)
    _style_report_axis(ax, "", x_label, y_label)


def _add_report_header(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    page_number: int,
    page_count: int,
) -> None:
    fig.text(
        0.055, 0.95, title, fontsize=17, fontweight="bold",
        color="#102a43", va="top",
    )
    fig.text(0.055, 0.895, subtitle, fontsize=8.5, color="#486581", va="top")
    fig.text(
        0.945, 0.045, f"MecaLab  •  Page {page_number}/{page_count}",
        fontsize=7, color="#627d98", ha="right",
    )
    fig.add_artist(
        Line2D([0.055, 0.945], [0.87, 0.87], transform=fig.transFigure,
               color="#176b87", linewidth=1.3)
    )


def build_analysis_pdf(
    curves: list[dict[str, object]],
    comparison: dict[str, object] | None,
    x_label: str,
    y_label: str,
) -> bytes:
    """Create an A4 landscape report with one page per curve and an optional comparison."""
    output = BytesIO()
    page_count = len(curves) + (1 if comparison is not None else 0)
    with PdfPages(
        output,
        metadata={
            "Title": "Rapport d'analyse mécanique — MecaLab",
            "Author": "MecaLab - Nemelion13",
            "Subject": "Courbes de traction/compression et comparaison",
        },
    ) as pdf:
        for page_number, curve in enumerate(curves, start=1):
            fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
            grid = fig.add_gridspec(
                2, 2, left=0.055, right=0.945, bottom=0.12, top=0.82,
                hspace=0.36, wspace=0.22, height_ratios=[1.0, 0.26],
            )
            filename = str(curve["filename"])
            label = str(curve["label"])
            configuration = str(curve["configuration"])
            _add_report_header(
                fig,
                f"Analyse de courbe — {label}",
                f"Fichier : {filename}   |   Configuration : {configuration}",
                page_number,
                page_count,
            )

            full_axis = fig.add_subplot(grid[0, 0])
            zoom_axis = fig.add_subplot(grid[0, 1])
            full_axis.set_title(
                "Courbe complète", loc="left", fontsize=10, fontweight="bold",
                color="#102a43",
            )
            zoom_axis.set_title(
                "Détail de la zone initiale (ε = 0 à 0,0035)",
                loc="left", fontsize=10, fontweight="bold", color="#102a43",
            )
            _plot_report_curve(full_axis, curve, x_label, y_label)
            _plot_report_curve(
                zoom_axis, curve, x_label, y_label, x_limits=(0.0, 0.0035)
            )

            metrics = curve["metrics"]
            assert isinstance(metrics, dict)
            metric_items = [
                ("Module d’Young", _report_value(metrics["Module d'Young (MPa)"], " MPa")),
                ("Re à ε max", _report_value(
                    metrics["Re à la borne élastique max. (MPa)"], " MPa"
                )),
                ("Rp0,2", _report_value(metrics["Rp0,2 (MPa)"], " MPa")),
                ("Rm", _report_value(metrics["Rm traction (MPa)"], " MPa")),
                ("Rc", _report_value(metrics["Rc compression (MPa)"], " MPa")),
                ("Points valides", str(metrics["Points valides"])),
            ]
            table_axis = fig.add_subplot(grid[1, :])
            table_axis.axis("off")
            table = table_axis.table(
                cellText=[[value for _, value in metric_items]],
                colLabels=[name for name, _ in metric_items],
                cellLoc="center",
                loc="center",
                bbox=[0, 0.15, 1, 0.75],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for (row, _), cell in table.get_celld().items():
                cell.set_edgecolor("#dce8ef")
                if row == 0:
                    cell.set_facecolor("#176b87")
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                else:
                    cell.set_facecolor("#f3f8fb")
                    cell.get_text().set_color("#102a43")
            elastic_range = curve["elastic_range"]
            table_axis.text(
                0.0, 0.02,
                f"Zone de régression élastique : ε ∈ "
                f"[{float(elastic_range[0]):.6g}, {float(elastic_range[1]):.6g}]. "
                "Re correspond à la contrainte interpolée à la borne supérieure choisie ; "
                "Rp0,2 est la limite conventionnelle décalée.",
                transform=table_axis.transAxes, fontsize=7, color="#486581",
            )
            pdf.savefig(fig)
            plt.close(fig)

        if comparison is not None:
            page_number = len(curves) + 1
            reference = comparison["reference"]
            compared = comparison["compared"]
            assert isinstance(reference, dict) and isinstance(compared, dict)
            fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
            grid = fig.add_gridspec(
                2, 2, left=0.055, right=0.945, bottom=0.12, top=0.82,
                hspace=0.38, wspace=0.22, height_ratios=[1.0, 0.26],
            )
            _add_report_header(
                fig,
                "Comparaison des courbes",
                f"Référence : {reference['label']} ({reference['filename']})   |   "
                f"Comparée : {compared['label']} ({compared['filename']})",
                page_number,
                page_count,
            )
            for axis, x_limits, chart_title in (
                (fig.add_subplot(grid[0, 0]), None, "Courbes complètes"),
                (
                    fig.add_subplot(grid[0, 1]),
                    (0.0, 0.0022),
                    "Détail de la zone initiale (ε = 0 à 0,0022)",
                ),
            ):
                _plot_report_curve(
                    axis, reference, x_label, y_label, x_limits, set_zoom_limits=False
                )
                _plot_report_curve(
                    axis, compared, x_label, y_label, x_limits, set_zoom_limits=False
                )
                if x_limits is not None:
                    _set_report_zoom_y_limits(axis, [reference, compared], x_limits)
                axis.set_title(
                    chart_title, loc="left", fontsize=10, fontweight="bold",
                    color="#102a43",
                )
                axis.axvspan(
                    float(comparison["strain_min"]),
                    float(comparison["strain_max"]),
                    color="#2a9d8f",
                    alpha=0.08,
                    label="Intervalle commun",
                )

            comparison_metrics = comparison["metrics"]
            assert isinstance(comparison_metrics, dict)
            metric_items = [
                ("RMSE", _report_value(comparison_metrics["rmse"], " MPa")),
                ("Écart absolu max.", _report_value(comparison_metrics["max_error"], " MPa")),
                ("Biais moyen", _report_value(comparison_metrics["mean_bias"], " MPa")),
                ("RMSE relatif", _report_value(comparison_metrics["relative_rmse"], " %")),
            ]
            table_axis = fig.add_subplot(grid[1, :])
            table_axis.axis("off")
            table = table_axis.table(
                cellText=[[value for _, value in metric_items]],
                colLabels=[name for name, _ in metric_items],
                cellLoc="center",
                loc="center",
                bbox=[0.06, 0.2, 0.88, 0.7],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            for (row, _), cell in table.get_celld().items():
                cell.set_edgecolor("#dce8ef")
                if row == 0:
                    cell.set_facecolor("#176b87")
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")
                else:
                    cell.set_facecolor("#f3f8fb")
                    cell.get_text().set_color("#102a43")
            table_axis.text(
                0.0, 0.02,
                f"Métriques évaluées sur 500 points interpolés dans l’intervalle commun "
                f"ε ∈ [{float(comparison['strain_min']):.6g}, "
                f"{float(comparison['strain_max']):.6g}]. "
                "Biais = contrainte comparée − contrainte de référence ; aucune extrapolation.",
                transform=table_axis.transAxes, fontsize=7, color="#486581",
            )
            pdf.savefig(fig)
            plt.close(fig)

    return output.getvalue()