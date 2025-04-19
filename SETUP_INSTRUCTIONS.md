# Setup Instructions for 3D Reconstruction Pipeline

## Prerequisites

Before running the setup script, make sure you have:

1. Python 3.7+ installed
   ```bash
   # Check Python version
   python3 --version
   ```

2. Python virtual environment package
   ```bash
   # Debian/Ubuntu
   sudo apt-get install python3-venv
   
   # macOS (with Homebrew)
   brew install python3
   
   # Windows
   # Should be included with Python installation
   ```

## Setup Process

1. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. If the virtual environment setup fails, you can manually create it:
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   
   # Activate it
   source .venv/bin/activate  # On Linux/macOS
   # OR
   .venv\Scripts\activate     # On Windows
   
   # Install dependencies
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```

3. Always make sure the virtual environment is activated before running the pipeline:
   ```bash
   source .venv/bin/activate  # On Linux/macOS
   # OR
   .venv\Scripts\activate     # On Windows
   ```

## Running the Pipeline

After setup is complete and the virtual environment is activated:

```bash
python3 scripts/run_pipeline.py --images data/sample --visualise
```

## Troubleshooting

- If you see `ModuleNotFoundError: No module named 'cv2'` or similar errors, the dependencies aren't installed correctly or the virtual environment isn't activated.

- Make sure your sample image directory (data/sample) contains images. The repository doesn't include images due to size constraints.

- For macOS users with Apple Silicon, you might need to install some packages with specific options:
  ```bash
  pip install --no-binary=opencv-python opencv-python
  ```

- If Open3D installation fails, try:
  ```bash
  pip install open3d --no-cache-dir
  ``` 