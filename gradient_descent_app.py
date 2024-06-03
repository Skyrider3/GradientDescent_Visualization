import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox, QDoubleSpinBox, QCheckBox
from PySide6.QtCore import Qt
from PySide6.QtWebEngineWidgets import QWebEngineView
from gradient_descent import *
from utils import *

class GradientDescentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Gradient Descent Viz")
        self.setGeometry(300, 300, 1200, 800)
        self.createWidgets()
        self.createLayout()

    def createWidgets(self):
        self.plot_area = QWidget()
        self.plot_area.setLayout(QVBoxLayout())
        self.sidebar = QWidget()
        self.sidebar.setLayout(QVBoxLayout())

        self.momentum_slider = QSlider(Qt.Horizontal)
        self.momentum_slider.setMinimum(0)
        self.momentum_slider.setMaximum(100)
        self.momentum_slider.setValue(80)  # Initial value (representing 0.8)
        self.batch_size_slider = QSlider(Qt.Horizontal)
        self.batch_size_slider.setMinimum(1)
        self.batch_size_slider.setMaximum(100)
        self.batch_size_slider.setValue(10)

        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.plot_widget = QWebEngineView()

        self.plot_area.layout().addWidget(self.plot_widget)

        self.sidebar.layout().addWidget(QLabel("Momentum:"))
        self.sidebar.layout().addWidget(self.momentum_slider)
        self.sidebar.layout().addWidget(QLabel("Batch Size:"))
        self.sidebar.layout().addWidget(self.batch_size_slider)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Gradient Descent", "Momentum", "Adagrad", "RMSprop", "Adam"])
        self.sidebar.layout().addWidget(QLabel("Optimizer:"))
        self.sidebar.layout().addWidget(self.optimizer_combo)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 1.0)
        self.learning_rate_spin.setValue(0.01)
        self.sidebar.layout().addWidget(QLabel("Learning Rate:"))
        self.sidebar.layout().addWidget(self.learning_rate_spin)

        self.decay_rate_spin = QDoubleSpinBox()
        self.decay_rate_spin.setRange(0.0, 1.0)
        self.decay_rate_spin.setValue(0.9)
        self.sidebar.layout().addWidget(QLabel("Decay Rate:"))
        self.sidebar.layout().addWidget(self.decay_rate_spin)
        self.beta1_spin = QDoubleSpinBox()
        self.beta1_spin.setRange(0.0, 1.0)
        self.beta1_spin.setValue(0.9)
        self.sidebar.layout().addWidget(QLabel("Beta1:"))
        self.sidebar.layout().addWidget(self.beta1_spin)
        self.beta2_spin = QDoubleSpinBox()
        self.beta2_spin.setRange(0.0, 1.0)
        self.beta2_spin.setValue(0.999)
        self.sidebar.layout().addWidget(QLabel("Beta2:"))
        self.sidebar.layout().addWidget(self.beta2_spin)

        self.show_gradient_arrows_checkbox = QCheckBox("Gradient Arrows")
        self.show_adjusted_gradient_arrows_checkbox = QCheckBox("Adjusted Gradient Arrows")
        self.show_momentum_arrows_checkbox = QCheckBox("Momentum Arrows")
        self.show_sum_of_gradient_squared_checkbox = QCheckBox("Sum of Gradient Squared")
        self.show_path_checkbox = QCheckBox("Path")

        self.sidebar.layout().addWidget(self.show_gradient_arrows_checkbox)
        self.sidebar.layout().addWidget(self.show_adjusted_gradient_arrows_checkbox)
        self.sidebar.layout().addWidget(self.show_momentum_arrows_checkbox)
        self.sidebar.layout().addWidget(self.show_sum_of_gradient_squared_checkbox)
        self.sidebar.layout().addWidget(self.show_path_checkbox)
        self.sidebar.layout().addWidget(self.play_button)
        self.sidebar.layout().addWidget(self.pause_button)
        self.sidebar.layout().addWidget(self.stop_button)

        self.momentum_slider.valueChanged.connect(self.update_plot)
        self.batch_size_slider.valueChanged.connect(self.update_plot)
        self.optimizer_combo.currentIndexChanged.connect(self.update_plot)
        self.learning_rate_spin.valueChanged.connect(self.update_plot)
        self.decay_rate_spin.valueChanged.connect(self.update_plot)
        self.beta1_spin.valueChanged.connect(self.update_plot)
        self.beta2_spin.valueChanged.connect(self.update_plot)
        self.show_gradient_arrows_checkbox.stateChanged.connect(self.update_plot)
        self.show_adjusted_gradient_arrows_checkbox.stateChanged.connect(self.update_plot)
        self.show_momentum_arrows_checkbox.stateChanged.connect(self.update_plot)
        self.show_sum_of_gradient_squared_checkbox.stateChanged.connect(self.update_plot)
        self.show_path_checkbox.stateChanged.connect(self.update_plot)

        self.update_plot()

    def createLayout(self):
        layout = QHBoxLayout()
        layout.addWidget(self.plot_area)
        layout.addWidget(self.sidebar)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def update_plot(self):
        learning_rate = self.learning_rate_spin.value()
        momentum = self.momentum_slider.value() / 100
        batch_size = self.batch_size_slider.value()
        optimizer = self.optimizer_combo.currentText()
        decay_rate = self.decay_rate_spin.value()
        beta1 = self.beta1_spin.value()
        beta2 = self.beta2_spin.value()

        show_gradient_arrows = self.show_gradient_arrows_checkbox.isChecked()
        show_adjusted_gradient_arrows = self.show_adjusted_gradient_arrows_checkbox.isChecked()
        show_momentum_arrows = self.show_momentum_arrows_checkbox.isChecked()
        show_sum_of_gradient_squared = self.show_sum_of_gradient_squared_checkbox.isChecked()
        show_path = self.show_path_checkbox.isChecked()

        x, y = 10, 10
        iterations = 100

        if optimizer == "Gradient Descent":
            x_trajectory, y_trajectory = gradient_descent(x, y, learning_rate, iterations)
        elif optimizer == "Momentum":
            x_trajectory, y_trajectory = momentum_gradient_descent(x, y, learning_rate, momentum, iterations)
        elif optimizer == "Adagrad":
            x_trajectory, y_trajectory = adagrad_gradient_descent(x, y, learning_rate, iterations)
        elif optimizer == "RMSprop":
            x_trajectory, y_trajectory = rmsprop_gradient_descent(x, y, learning_rate, decay_rate, iterations)
        elif optimizer == "Adam":
            x_trajectory, y_trajectory = adam_gradient_descent(x, y, learning_rate, beta1, beta2, iterations)

        fig = create_visualization(x_trajectory, y_trajectory, optimizer, show_gradient_arrows, show_path)
        self.plot_widget.setHtml(fig.to_html(include_plotlyjs='cdn'))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GradientDescentApp()
    window.show()
    sys.exit(app.exec())