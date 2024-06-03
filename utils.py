import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
from gradient_descent import *

def create_visualization(
    x_trajectory: List[float],
    y_trajectory: List[float],
    optimizer: str,
    show_arrows: bool = False,
    show_path: bool = False
) -> go.Figure:
    x_range = np.linspace(-10, 10, 100)
    y_range = np.linspace(-10, 10, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = cost_function(x_grid, y_grid)

    z_trajectory = [cost_function(x, y) for x, y in zip(x_trajectory, y_trajectory)]

    fig = go.Figure(data=[
        go.Surface(x=x_range, y=y_range, z=z_grid),
        go.Scatter3d(
            x=x_trajectory,
            y=y_trajectory,
            z=z_trajectory,
            mode='lines+markers',
            name=optimizer
        )
    ])

    if show_arrows:
        add_gradient_arrows(fig, x_trajectory, y_trajectory)

    if show_path:
        add_path(fig, x_trajectory, y_trajectory, z_trajectory)

    set_layout(fig)
    return fig

def add_gradient_arrows(
    fig: go.Figure,
    x_trajectory: List[float],
    y_trajectory: List[float]
) -> None:
    for x, y in zip(x_trajectory, y_trajectory):
        dx, dy = gradient(x, y)
        fig.add_trace(go.Cone(
            x=[x], y=[y], z=[cost_function(x, y)],
            u=[dx], v=[dy], w=[0],
            colorscale='Viridis',
            name='Gradient'
        ))

def add_path(
    fig: go.Figure,
    x_trajectory: List[float],
    y_trajectory: List[float],
    z_trajectory: List[float]
) -> None:
    fig.add_trace(go.Scatter3d(
        x=x_trajectory,
        y=y_trajectory,
        z=z_trajectory,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Path'
    ))

def set_layout(fig: go.Figure) -> None:
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Cost',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        title='Gradient Descent Visualization',
        showlegend=True
    )

def calculate_cost_trajectory(
    x_trajectory: List[float],
    y_trajectory: List[float]
) -> List[float]:
    return [cost_function(x, y) for x, y in zip(x_trajectory, y_trajectory)]

def create_meshgrid(
    start: float = -10,
    end: float = 10,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_range = np.linspace(start, end, num_points)
    y_range = np.linspace(start, end, num_points)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = cost_function(x_grid, y_grid)
    return x_grid, y_grid, z_grid

if __name__ == "__main__":
    # Example usage for testing
    x, y = 5, 5
    iterations = 50
    x_traj, y_traj = gradient_descent(x, y, 0.1, iterations)
    fig = create_visualization(x_traj, y_traj, "Gradient Descent", show_arrows=True, show_path=True)
    fig.show()
