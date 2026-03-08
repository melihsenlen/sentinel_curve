# sentinel_curve
A resource monitoring & prediction tool that collects CPU/memory usage, feeds it into a <a href="https://pytorch.org/">PyTorch</a> regression model that predicts future resource usage interactively.

## Features
- Real-time CPU and memory monitoring via C++ (Windows)
- Time-series regression using LSTM in Python (PyTorch)
- Interactive predictions and plots in Jupyter Notebook
- Comes with a Batch file to make things easier

## Prequisites
- Windows
- C++ compiler
- Python 3.10+
- Torch
- Pandas
- Numpy
- Matplotlib
- Scikit-learn
- Ipywidgets
- Ipython

## Usage
1. Run the pipeline:
    ```bash
    cd pipeline
    run.bat
    ```
    This will create <code>data/data.csv</code> containing CPU (%) and memory (MB) usage.

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. Open <code>notebooks/exploration.ipynb</code>:
    - Run cells sequentially to train the model and see predicted CPU/memory trends

## Arguments
You can customize how the monitor runs by passing arguments to the batch script:

| Argument  | Default | Description |
|-----------|---------|-------------|
| <code>--interval</code> | 1 | Time interval (seconds) between each measurement |
| <code>--duration</code> | 60 | Total duration (seconds) to run the monitor |
| <code>--output</code> | <code>data/data.csv</code> | Path to the output CSV file where measurements are saved |

## License
MIT License