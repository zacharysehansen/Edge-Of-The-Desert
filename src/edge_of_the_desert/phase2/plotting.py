from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_historical_sustainability_plot(
    csv_path: str | Path,
    output_path: str | Path,
    model_label: str | None = None,
) -> Path:
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    historical = pd.read_csv(csv_path)
    historical_plot_x = pd.to_datetime(historical["year_month"])

    fig, ax = plt.subplots(figsize=(12, 5))

    y_col = "usdm_sustainability"
    line_label = "Sustainability"
    if "actual_usdm_sustainability" in historical.columns:
        line_label = (
            f"Predicted ({model_label})" if model_label else "Predicted sustainability"
        )

    ax.plot(
        historical_plot_x,
        historical[y_col],
        color="#1f5aa6",
        linewidth=2.0,
        label=line_label,
    )

    if "actual_usdm_sustainability" in historical.columns:
        ax.scatter(
            historical_plot_x,
            historical["actual_usdm_sustainability"],
            color="#d4772a",
            s=14,
            alpha=0.55,
            label="Actual USDM sustainability",
        )
    else:
        ax.scatter(
            historical_plot_x,
            historical[y_col],
            color="#1f5aa6",
            s=14,
            alpha=0.55,
            label="Saved sustainability",
        )

    ax.set_title("Historical Sustainability Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sustainability")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
