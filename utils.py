import numpy as np
import plotly.graph_objects as go
from gradient_descent import *

def create_visualization(x_trajectory, y_trajectory, optimizer, show_arrows=False, show_path=False):
    fig = go.Figure(data=[
        go.Surface(x=np.linspace(-10, 10, 100),
                   y=np.linspace(-10, 10, 100),
                   z=cost_function(np.linspace(-10, 10, 100)[:, None],
                                   np.linspace(-10, 10, 100)[None, :])),
        go.Scatter3d(x=x_trajectory,
                     y=y_trajectory,
                     z=[cost_function(x, y) for x, y in zip(x_trajectory, y_trajectory)],
                     mode='lines+markers')
    ])
    if show_arrows:
        for x, y in zip(x_trajectory, y_trajectory):
            dx, dy = gradient(x, y)
            fig.add_trace(go.Cone(x=[x], y=[y], u=[dx], v=[dy], colorscale='Viridis'))
    if show_path:
        fig.add_trace(go.Scatter3d(x=x_trajectory, y=y_trajectory, z=[cost_function(x, y) for x, y in zip(x_trajectory, y_trajectory)], mode='lines', line=dict(color='blue')))
    return fig
