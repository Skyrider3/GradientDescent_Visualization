import numpy as np
import plotly.graph_objects as go

def cost_function(x: float, y: float) -> float:
    return x**2 + y**2

def gradient(x: float, y: float) -> tuple[float, float]:
    return 2*x, 2*y

def gradient_descent(x: float, y: float, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    x_trajectory = [x]
    y_trajectory = [y]
    for _ in range(iterations):
        dx, dy = gradient(x, y)
        x -= learning_rate * dx
        y -= learning_rate * dy
        x_trajectory.append(x)
        y_trajectory.append(y)
    return x_trajectory, y_trajectory

def momentum_gradient_descent(x: float, y: float, learning_rate: float, momentum: float, iterations: int) -> tuple[list[float], list[float]]:
    v_x = 0.0
    v_y = 0.0
    x_trajectory = [x]
    y_trajectory = [y]
    for _ in range(iterations):
        dx, dy = gradient(x, y)
        v_x = momentum * v_x + learning_rate * dx
        v_y = momentum * v_y + learning_rate * dy
        x -= v_x
        y -= v_y
        x_trajectory.append(x)
        y_trajectory.append(y)
    return x_trajectory, y_trajectory

def adagrad_gradient_descent(x: float, y: float, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    g_x = 0.0
    g_y = 0.0
    x_trajectory = [x]
    y_trajectory = [y]
    for _ in range(iterations):
        dx, dy = gradient(x, y)
        g_x += dx**2
        g_y += dy**2
        x -= learning_rate * dx / (np.sqrt(g_x) + 1e-8)
        y -= learning_rate * dy / (np.sqrt(g_y) + 1e-8)
        x_trajectory.append(x)
        y_trajectory.append(y)
    return x_trajectory, y_trajectory

def rmsprop_gradient_descent(x: float, y: float, learning_rate: float, decay_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    g_x = 0.0
    g_y = 0.0
    x_trajectory = [x]
    y_trajectory = [y]
    for _ in range(iterations):
        dx, dy = gradient(x, y)
        g_x = decay_rate * g_x + (1 - decay_rate) * dx**2
        g_y = decay_rate * g_y + (1 - decay_rate) * dy**2
        x -= learning_rate * dx / (np.sqrt(g_x) + 1e-8)
        y -= learning_rate * dy / (np.sqrt(g_y) + 1e-8)
        x_trajectory.append(x)
        y_trajectory.append(y)
    return x_trajectory, y_trajectory

def adam_gradient_descent(x: float, y: float, learning_rate: float, beta1: float, beta2: float, iterations: int) -> tuple[list[float], list[float]]:
    m_x = 0.0
    m_y = 0.0
    v_x = 0.0
    v_y = 0.0
    x_trajectory = [x]
    y_trajectory = [y]
    for i in range(iterations):
        dx, dy = gradient(x, y)
        m_x = beta1 * m_x + (1 - beta1) * dx
        m_y = beta1 * m_y + (1 - beta1) * dy
        v_x = beta2 * v_x + (1 - beta2) * dx**2
        v_y = beta2 * v_y + (1 - beta2) * dy**2
        m_x_hat = m_x / (1 - beta1**(i + 1))
        m_y_hat = m_y / (1 - beta1**(i + 1))
        v_x_hat = v_x / (1 - beta2**(i + 1))
        v_y_hat = v_y / (1 - beta2**(i + 1))
        x -= learning_rate * m_x_hat / (np.sqrt(v_x_hat) + 1e-8)
        y -= learning_rate * m_y_hat / (np.sqrt(v_y_hat) + 1e-8)
        x_trajectory.append(x)
        y_trajectory.append(y)
    return x_trajectory, y_trajectory