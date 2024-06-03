# Looking for Contributors
# This is open source and open for everyone 
* Actively looking for contributors to help improve this project. If you are interested in contributing, please reach out to me [Linkedin] : (https://www.linkedin.com/in/umamaheswar-e-innovator/).I are particularly interested in contributions that can help add web visualization, including 3D gradient visualization and 3D animation.



# Gradient Descent Visualization

This repository contains a Python application that visualizes the gradient descent algorithm for various optimization problems. The app allows users to:

* Select different optimization algorithms (Gradient Descent, Momentum, Adagrad, RMSprop, Adam)
* Adjust the learning rate, momentum, and other parameters
* Visualize the cost function surface
* See the trajectory of the optimization process
* Add gradient arrows, adjusted gradient arrows, momentum arrows, and the sum of squared gradients

* First , Create virtual environemnt on your local machine

```sh
python -m venv/env-name
```
* Now activate virtual environment

```sh
source env-name/bin/activate
```

## Installation

1. Clone this repository:
```sh
git clone https://github.com/Skyrider3/GradientDescent_Visualization.git
```
2. Install the required dependencies:
```sh
pip install -r requirements.txt
```
## Usage

1. Run the application:
```sh
python gradient_descent_app.py
```
2. A window will appear with a sidebar and a plot area.
3. Use the sliders and checkboxes in the sidebar to adjust the parameters and options.
4. The plot area will show the cost function surface, the optimization trajectory, and any selected visualization elements.

## Features

* **Multiple optimization algorithms:** Choose from Gradient Descent, Momentum, Adagrad, RMSprop, and Adam.
* **Adjustable parameters:** Modify the learning rate, momentum, decay rate, beta1, beta2, and other parameters.
* **Interactive visualization:** See the cost function surface, the optimization trajectory, and various visualization elements.
* **Gradient arrows:** Visualize the gradient at each point in the trajectory.
* **Adjusted gradient arrows:** See the adjusted gradient for algorithms like Momentum, RMSprop, and Adam.
* **Momentum arrows:** Visualize the momentum vector in the Momentum algorithm.
* **Sum of squared gradients:** See the sum of squared gradients for each point in the trajectory.
* **Path:** See the path taken by the optimization process.

## Visual

![alt text](https://github.com/Skyrider3/GradientDescent_Visualization/blob/main/Image.png)

## Contributing

Contributions are welcome! Please see the [contribution guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.