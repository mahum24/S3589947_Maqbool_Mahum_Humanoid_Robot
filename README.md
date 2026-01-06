


#### **Humanoid Reinforcement Learning Project**

Reinforcement learning project implemented using Stable-Baselines3 to train a humanoid agent in the Gymnasium environment with TensorBoard visualization.

#### 📋 **Prerequisites**



#### **macOS:**
```bash
brew install ffmpeg
```

#### **Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install ffmpeg
```

#### **Windows:**
Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH, or use:
```bash
choco install ffmpeg  # If using Chocolatey
```

### Python Version
- Python 3.8 or higher recommended

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/mahum24/S3589947_Maqbool_Mahum_Humanoid_Robot.git
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Note:** If you encounter installation issues, try upgrading pip first:
```bash
pip install --upgrade pip
```

### 3. Run the Project
```bash
python humanoid_rl_project.py
```

## 📁 Project Structure
```
humanoid_rl_project/
├── humanoid_rl_project.py                 # Main training script
├── requirements.txt                       # Python dependencies
├── logs/                                  # TensorBoard logs (auto-created)
│   └── [algorithm_name]/                  # Algorithm-specific logs
├── models/                                # Saved models (auto-created)
│   └── [algorithm_name]/                  # Algorithm-specific models
├── plots/                                 # Save Plots
├── results/                               # Save Json Result and txt and detailed read me of all project
│   └── [algorithm_name_results.json/]     # Algorithm-specific json
│   └── [project_report_time.md]           # Whole project report
│   └── [algorithm_name_visualization_report.txt]     # Algorithm descriptive report
├── videos/                                # Saved Videos
│   └── [algorithm_name]/                  # Algorithm evaluation video 
└── README.md                              # This file
```

## 🏃 How to Use

### Running the Project
```bash
python humanoid_rl_project.py
```

One starting script have several options:
1. Display Environment Information
2. Train a New Model
3. Load and Test a Trained Model
4. Visualize Model Performance
5. Evaluate Model Quantitatively
6. Compare Multiple Algorithms
7. Display Available Trained Models
8. Run Complete Experiment Pipeline
9. Generate Project Report
0. Exit

### Monitoring Training
To view TensorBoard logs during/after training:
```bash
tensorboard --logdir=logs/
```

Then open your browser and navigate to:
```
http://localhost:6006
```

## 🧠 Available Algorithms

The project supports two RL algorithms. Models and logs are saved in algorithm-specific directories with the same name:

- **PPO** - Proximal Policy Optimization (`logs/ppo/`, `models/ppo/`)
- **SAC** - Soft Actor-Critic (`logs/sac/`, `models/sac/`)

## 💾 Model Management

### Best Model Location
The best model is automatically saved during training at:
```
trained_models/[algorithm_name]/best_model.zip
```

For example:
- PPO algorithm: `trained_models/ppo/best_model.zip`
- SAC algorithm: `trained_models/sac/best_model.zip`

### Loading a Trained Model
Select the option 3 from the Menu

If using Best Model It will ask for Name
Give the Name Exact Above of that from Directory


## ⚙️ Configuration

You can modify training parameters in `humanoid_rl_project.py`:

- **Environment**: `Humanoid-v4` (Gymnasium)
- **Training timesteps**: Default 1,000,000
- **Evaluation frequency**: Every 50,000 timesteps
- **Video recording**: Enabled for evaluation episodes

## 📊 Metrics Tracked in TensorBoard

The following metrics are logged during training:
- `episode_reward`: Total reward per episode
- `episode_length`: Steps per episode
- `value_loss`: Value function loss
- `policy_loss`: Policy gradient loss
- `entropy_loss`: Policy entropy (exploration)
- `learning_rate`: Current learning rate

## 🔧 Troubleshooting

### Common Issues

1. **FFmpeg not found error:**
   ```
   RuntimeError: Could not find ffmpeg
   ```
   **Solution:** Install FFmpeg system dependency (see Prerequisites section).

2. **PyTorch/CUDA issues:**
   ```bash
   # Install CPU-only version if GPU issues occur
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Missing dependencies:**
   ```bash
   # Install any missing packages manually
   pip install imageio[ffmpeg] Pillow
   ```

4. **Virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   pip install -r requirements.txt
   ```


