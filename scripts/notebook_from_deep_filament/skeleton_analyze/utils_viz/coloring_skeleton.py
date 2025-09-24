import numpy as np
import plotly.graph_objs as go
import plotly.express as px

def create_clustered_scatter_plot(points, labels, jitter_strength=0.2):
    points = np.array(points)

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    discrete_colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    discrete_colors = discrete_colors[:num_classes]  

    label_to_color = {label: discrete_colors[i] for i, label in enumerate(unique_labels)}

        
    # Initialize data and dictionary for jittered points per label
    scatter_data = []
    seen_points = {}  # Track seen points for jittering

    # Collect points by label to group them in the plot
    points_by_label = {}

    # Collect points and jitter if necessary
    for point, label in zip(points, labels):
        if label not in points_by_label:
            points_by_label[label] = []

        # Add a slight jitter if this point has already been seen
        if tuple(point) in seen_points:
            jitter = np.random.uniform(-jitter_strength, jitter_strength, size=2)
            jittered_point = point + jitter
        else:
            jittered_point = point
            seen_points[tuple(point)] = True  # Mark as seen

        points_by_label[label].append(jittered_point)

    # Create one trace per label for the legend
    for label, jittered_points in points_by_label.items():
        x_vals = [pt[0] for pt in jittered_points]
        y_vals = [pt[1] for pt in jittered_points]

        scatter_data.append(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(
                size=10,
                color=label_to_color[label],
                showscale=False
            ),
            name=f"Label {label}"  # Only one entry per label in the legend
        ))

    # Define layout
    layout = go.Layout(
        title="Clustered Skeleton Points (with Jittering)",
        xaxis=dict(title="X"),
        yaxis=dict(title="Y", autorange='reversed'),
        height=800,
        width=800
    )

    # Create the figure
    fig = go.Figure(data=scatter_data, layout=layout)

    return fig
