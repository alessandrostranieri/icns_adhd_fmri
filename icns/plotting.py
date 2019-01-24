import matplotlib.pyplot as plt
import numpy as np


def plot_institute_scores(institute_scores: dict, filename=None) -> None:
    # Parameters and style
    width = 1.0
    metric_colors = {0: 'black',  # Accuracy
                     1: 'red',  # Precision
                     2: 'blue',
                     3: 'green'}  # Recall

    metric_labels = {0: 'Accuracy',
                     1: 'Precision',
                     2: 'Recall',
                     3: 'Chance'}

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.grid(True, axis='y', zorder=0)

    bar_positions = list()
    legend_handles = [None] * 4
    for pos, institute in enumerate(institute_scores.keys()):
        single_institute_results = institute_scores[institute]
        results_center_position = pos * 1.5 * width * len(legend_handles)
        bar_positions.append(results_center_position)
        for (metric_index, (metric_name, metric_value)) in enumerate(single_institute_results.items()):
            results_offset = results_center_position + (metric_index - 1)
            legend_handles[metric_index] = ax.bar(results_offset, metric_value, width=width, bottom=0,
                                                  color=metric_colors[metric_index],
                                                  label=metric_labels[metric_index],
                                                  zorder=3)

    ax.set_title('Classification Scores by Institute')
    ax.set_ylabel('Score')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Institutes')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(institute_scores.keys(), rotation=45)

    plt.legend(legend_handles, list(metric_labels.values()))

    if filename:
        plt.savefig(filename)

    plt.show()
