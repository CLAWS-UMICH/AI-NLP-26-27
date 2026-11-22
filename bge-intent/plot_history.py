"""Render training history JSON as dependency-free SVG charts."""

import json
from pathlib import Path


def chart(epochs, series, title, ylabel, destination):
    width, height = 760, 460
    left, right, top, bottom = 78, 28, 54, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [value for _, values in series for value in values]
    low, high = min(0.0, min(values)), max(values)
    if high == low:
        high = low + 1
    pad = (high - low) * 0.08
    low, high = low - pad, high + pad
    x_span = max(1, len(epochs) - 1)
    x = lambda index: left + index * plot_w / x_span
    y = lambda value: top + (high - value) * plot_h / (high - low)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">{title}</text>']
    for tick in range(6):
        value = low + (high - low) * tick / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.3g}</text>')
    for index, epoch in enumerate(epochs):
        xx = x(index)
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="11">{epoch}</text>')
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
              f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
              f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="13">Epoch</text>',
              f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>']
    colors = ["#2563eb", "#dc2626", "#059669"]
    for series_index, (label, values) in enumerate(series):
        points = " ".join(f'{x(i):.1f},{y(value):.1f}' for i, value in enumerate(values))
        color = colors[series_index % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for i, value in enumerate(values):
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(value):.1f}" r="4" fill="{color}"/>')
    legend_x = left
    for series_index, (label, _) in enumerate(series):
        color = colors[series_index % len(colors)]
        parts += [f'<line x1="{legend_x}" y1="{height-35}" x2="{legend_x+22}" y2="{height-35}" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{legend_x+28}" y="{height-31}" font-family="Arial" font-size="12">{label}</text>']
        legend_x += 170
    parts.append('</svg>')
    destination.write_text("\n".join(parts), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    metrics_path = args.run_dir / "metrics.json"
    history = json.loads(metrics_path.read_text(encoding="utf-8"))["history"]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    chart(epochs, [("Training loss", [row["train_loss"] for row in history]),
                   ("Validation loss", [row["validation_loss"] for row in history])],
          "Training and Validation Loss", "Cross-entropy loss", args.run_dir / "loss_curves.svg")
    chart(epochs, [("Validation accuracy", [row["validation_accuracy"] for row in history]),
                   ("Validation macro-F1", [row["validation_macro_f1"] for row in history])],
          "Validation Learning Curves", "Score", args.run_dir / "learning_curves.svg")


if __name__ == "__main__":
    main()
