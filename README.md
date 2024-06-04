# Setting Up a Python Virtual Environment (venv)

This guide will walk you through the process of setting up a Python virtual environment (venv) for the project.

## Prerequisites

- Python 3.x installed on your system
- pip (Python package installer) installed

## Steps to Set Up a Virtual Environment

### 1. Check Python Installation

Ensure you have Python installed by running the following command:

```sh
python --version
```

or, if you have multiple versions of Python installed:

```sh
python3 --version
```

### 2. Install `venv` (if not already installed)

The `venv` module is included in Python 3.3 and above. If it's not installed, you can install it using:

```sh
pip install virtualenv
```

### 3. Create a Virtual Environment

Navigate to your project directory and create a virtual environment by running:

```sh
python -m venv env
```

or, if you are using `python3`:

```sh
python3 -m venv env
```

This command creates a directory named `env` that contains the virtual environment.

### 4. Activate the Virtual Environment

#### On Windows

```sh
.\env\Scripts\activate
```

#### On macOS and Linux

```sh
source env/bin/activate
```

After activation, you should see `(env)` prefixed to your command prompt, indicating that the virtual environment is active.

### 5. Install Dependencies

With the virtual environment activated, you can now install any project dependencies. For example:

```sh
pip install <package_name>
```

### 6. Deactivate the Virtual Environment

To deactivate the virtual environment, simply run:

```sh
deactivate
```

## Additional Necessities

### Using a `requirements.txt` File

To create a `requirements.txt` file that lists all installed packages, run:

```sh
pip freeze > requirements.txt
```

To install the dependencies listed in a `requirements.txt` file:

```sh
pip install -r requirements.txt
```

### Virtual Environment Naming

You can name your virtual environment anything you like. For example, you might use `venv`, `myenv`, or any other descriptive name.

# Running Sypflix

Now, to access the WebApp, simply run

```sh
python app.py
```

or, if you are using `python3`:

```sh
python3 app.py
```

After doing the steps above, the website should be locally deployed in

```sh
localhost:3000
```