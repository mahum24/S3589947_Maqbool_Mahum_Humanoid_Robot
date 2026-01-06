"""
Humanoid-v5 Reinforcement Learning AI Project
"""
import gymnasium as gym
import numpy as np
import torch
import os
import sys
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import RL libraries
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    from stable_baselines3.common.logger import configure
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "gymnasium[box2d]", "stable-baselines3[extra]", 
                          "torch", "imageio", "opencv-python", "tensorboard"])
    print("Please restart the script.")
    sys.exit(1)

# Import visualization libraries
try:
    import imageio
    import cv2
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "imageio", "opencv-python", "tqdm"])
    import imageio
    import cv2
    from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class Config:
    """Configuration class for the project"""
    def __init__(self):
        self.project_root = Path("humanoid_rl_project")
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.videos_dir = self.project_root / "videos"
        self.results_dir = self.project_root / "results"
        self.configs_dir = self.project_root / "configs"
        
        # Create directories
        for dir_path in [self.models_dir, self.logs_dir, self.videos_dir, 
                        self.results_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Training parameters
        self.total_timesteps = 500000 
        self.eval_freq = 10000
        self.save_freq = 50000
        
        # Algorithm hyperparameters
        self.hyperparams = {
            'PPO': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.001,
                'policy_kwargs': dict(activation_fn=torch.nn.ReLU, net_arch=[400, 300])
            },
            'SAC': {
                'learning_rate': 1e-3,
                'buffer_size': 1000000,
                'learning_starts': 10000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'policy_kwargs': dict(activation_fn=torch.nn.ReLU, net_arch=[400, 300])
            }
        }

class HumanoidEnvironment:
    """Wrapper for the Humanoid-v5 environment"""
    
    @staticmethod
    def make_env(render_mode=None, seed=SEED):
        if render_mode:
            env = gym.make('Humanoid-v5', render_mode=render_mode)
        else:
            env = gym.make('Humanoid-v5')
        
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    
    @staticmethod
    def print_env_info():
        """Print information about the environment"""
        env = gym.make('Humanoid-v5')
        print("\n" + "="*60)
        print("HUMANOID-V5 ENVIRONMENT INFORMATION")
        print("="*60)
        print(f"Observation space: {env.observation_space}")
        print(f"Observation shape: {env.observation_space.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Action shape: {env.action_space.shape}")
        print(f"Action range: [{env.action_space.low.min():.2f}, {env.action_space.high.max():.2f}]")
        
        # Test reset
        obs, _ = env.reset()
        print(f"\nSample observation shape: {obs.shape}")
        print(f"First 5 observation values: {obs[:5]}")
        
        # Test action
        action = env.action_space.sample()
        print(f"Sample action shape: {action.shape}")
        print(f"First 3 action values: {action[:3]}")
        
        env.close()
        print("="*60 + "\n")

class HumanoidTrainer:
    """Trainer class for different RL algorithms"""
    
    def __init__(self, algorithm='PPO', config=None):
        self.algorithm = algorithm
        self.config = config or Config()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create algorithm-specific directories
        self.algorithm_dir = self.config.models_dir / f"{algorithm}_{self.timestamp}"
        self.algorithm_log_dir = self.config.logs_dir / f"{algorithm}_{self.timestamp}"
        self.algorithm_dir.mkdir(exist_ok=True)
        self.algorithm_log_dir.mkdir(exist_ok=True)
        
        # Initialize environments
        self.env = DummyVecEnv([lambda: HumanoidEnvironment.make_env()])
        self.eval_env = DummyVecEnv([lambda: HumanoidEnvironment.make_env()])
        
        self.model = None
        self.training_history = []
        
    def create_model(self):
        """Create model based on algorithm"""
        hyperparams = self.config.hyperparams[self.algorithm]
        
        print(f"\nCreating {self.algorithm} model with hyperparameters:")
        for key, value in hyperparams.items():
            if key != 'policy_kwargs':
                print(f"  {key}: {value}")
        
        if self.algorithm == 'PPO':
            model = PPO(
                'MlpPolicy',
                self.env,
                verbose=0,
                tensorboard_log=str(self.algorithm_log_dir),
                **hyperparams
            )
        elif self.algorithm == 'SAC':
            model = SAC(
                'MlpPolicy',
                self.env,
                verbose=0,
                tensorboard_log=str(self.algorithm_log_dir),
                **hyperparams
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return model
    
    def train(self):
        """Train the model"""
        print(f"\n{'='*60}")
        print(f"TRAINING {self.algorithm}")
        print(f"{'='*60}")
        
        self.model = self.create_model()
        
        # Create callbacks
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.algorithm_dir / "best_model"),
            log_path=str(self.algorithm_dir / "eval_logs"),
            eval_freq=self.config.eval_freq,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=str(self.algorithm_dir),
            name_prefix=self.algorithm
        )
        
        # Training progress callback
        class ProgressCallback(CallbackList):
            def __init__(self, callbacks, trainer):
                super().__init__(callbacks)
                self.trainer = trainer
                self.trainer.training_history = []
            
            def _on_step(self) -> bool:
                if self.num_timesteps % 1000 == 0:
                    progress = (self.num_timesteps / self.trainer.config.total_timesteps) * 100
                    print(f"Progress: {progress:.1f}% ({self.num_timesteps}/{self.trainer.config.total_timesteps})")
                return True
        
        callbacks = [eval_callback, checkpoint_callback]
        
        # Start training
        print(f"Starting training for {self.config.total_timesteps} timesteps...")
        print(f"Model will be saved to: {self.algorithm_dir}")
        print(f"Tensorboard logs: {self.algorithm_log_dir}")
        
        try:
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                tb_log_name=f"{self.algorithm}_run",
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user!")
        
        # Save final model
        final_model_path = self.algorithm_dir / f"{self.algorithm}_final"
        self.model.save(str(final_model_path))
        print(f"\nFinal model saved to: {final_model_path}.zip")
        
        # Save training configuration
        config_data = {
            'algorithm': self.algorithm,
            'timestamp': self.timestamp,
            'total_timesteps': self.config.total_timesteps,
            'hyperparameters': self.config.hyperparams[self.algorithm],
            'model_path': str(final_model_path)
        }
        
        config_file = self.algorithm_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Training configuration saved to: {config_file}")
        
        return self.model
    
    def load_model(self, model_path):
        """Load a trained model"""
        print(f"\nLoading model from: {model_path}")
        
        if self.algorithm == 'PPO':
            self.model = PPO.load(model_path)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        print(f"Model loaded successfully!")
        return self.model

class HumanoidVisualizer:
    """Visualization and evaluation class"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        
    def visualize_trained_model(self, model_path, algorithm, num_episodes=2, 
                                render_mode='rgb_array', save_video=True):
       
        """Visualize a trained model in action"""
        print(f"\n{'='*60}")
        print(f"VISUALIZING {algorithm} MODEL")
        print(f"{'='*60}")
        
        # Load model
        if algorithm == 'PPO':
            model = PPO.load(model_path)
        elif algorithm == 'SAC':
            model = SAC.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Create environment with rendering
        env = gym.make('Humanoid-v5', render_mode=render_mode)
        
        all_episodes_data = []
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            obs, _ = env.reset()
            frames = []
            episode_reward = 0
            episode_length = 0
            positions = []
            
            # Maximum steps per episode
            max_steps = 500
            
            for step in tqdm(range(max_steps), desc="Running episode"):
                action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                try:
                    frame = env.render()
                    if frame is not None:
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            frames.append(frame)
                        else:
                            frames.append(frame)
                except:
                    placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"Step: {step}", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    frames.append(placeholder)
                
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'data'):
                    positions.append(env.unwrapped.data.qpos[0].copy())
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    print(f"  Episode ended: {'terminated' if terminated else 'truncated'}")
                    break
            
            # Calculate forward distance
            forward_distance = 0
            if len(positions) > 1:
                forward_distance = positions[-1] - positions[0]
            
            episode_data = {
                'episode': episode + 1,
                'reward': float(episode_reward),  
                'length': int(episode_length),    
                'forward_distance': float(forward_distance), 
                'frames': frames
            }
            
            all_episodes_data.append(episode_data)
            
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Episode Length: {episode_length}")
            print(f"  Forward Distance: {forward_distance:.2f} meters")
            
            # Save video if requested and frames exist
            if save_video and frames:
                video_filename = f"{algorithm}_episode_{episode+1}_{datetime.now().strftime('%H%M%S')}.mp4"
                video_path = self.config.videos_dir / video_filename
                
                try:
                    # Check if imageio has ffmpeg
                    try:
                        writer = imageio.get_writer(str(video_path), fps=30, codec='libx264')
                    except Exception as e:
                        print(f"  FFMPEG not available: {e}")
                        print("  Trying alternative video saving method...")
                        
                        # Fallback: Use OpenCV to save video
                        if frames:
                            height, width = frames[0].shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (width, height))
                            
                            for frame in frames:
                                if frame.dtype != np.uint8:
                                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                                # Ensure frame is BGR for OpenCV
                                if len(frame.shape) == 3 and frame.shape[2] == 3:
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    video_writer.write(frame_bgr)
                            
                            video_writer.release()
                            print(f"  Video saved using OpenCV: {video_path}")
                        continue
                    
                    # Save frames using imageio
                    for frame in frames:
                        if frame.dtype != np.uint8:
                            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                        writer.append_data(frame)
                    writer.close()
                    
                    print(f"  Video saved to: {video_path}")
                    
                except Exception as e:
                    print(f"  Could not save video: {e}")
                    print("\n  To enable video recording, install:")
                    print("    1. imageio[ffmpeg]: pip install imageio[ffmpeg]")
                    print("    2. ffmpeg: brew install ffmpeg (Mac) or sudo apt-get install ffmpeg (Linux)")
        
        env.close()
        
        # Create a summary report
        self._create_visualization_report(all_episodes_data, algorithm)
        
        return all_episodes_data
    
    def _create_visualization_report(self, episodes_data, algorithm):
        """Create a visualization report"""
        report_path = self.config.results_dir / f"{algorithm}_visualization_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"{algorithm} Visualization Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for episode_data in episodes_data:
                f.write(f"Episode {episode_data['episode']}:\n")
                f.write(f"  Total Reward: {episode_data['reward']:.2f}\n")
                f.write(f"  Episode Length: {episode_data['length']}\n")
                f.write(f"  Forward Distance: {episode_data['forward_distance']:.2f} meters\n")
                f.write("\n")
        
        print(f"\nVisualization report saved to: {report_path}")
    
    def evaluate_model(self, model_path, algorithm, num_episodes=10):
        """Quantitatively evaluate a trained model"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {algorithm} MODEL")
        print(f"{'='*60}")
        
        # Load model
        if algorithm == 'PPO':
            model = PPO.load(model_path)
        elif algorithm == 'SAC':
            model = SAC.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        env = HumanoidEnvironment.make_env()
        
        metrics = {
            'rewards': [],
            'episode_lengths': [],
            'forward_distances': [],
            'energy_efficiency': []
        }
        
        print(f"Running {num_episodes} evaluation episodes...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            positions = []
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track position
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'data'):
                    positions.append(env.unwrapped.data.qpos[0].copy())
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # Calculate metrics
            forward_distance = 0
            if len(positions) > 1:
                forward_distance = positions[-1] - positions[0]
            
            energy_efficiency = forward_distance / (episode_length + 1e-6)
            
            metrics['rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)
            metrics['forward_distances'].append(forward_distance)
            metrics['energy_efficiency'].append(energy_efficiency)
            
            if (episode + 1) % 5 == 0:
                print(f"  Completed {episode + 1}/{num_episodes} episodes")
        
        env.close()
        
        stats = {}
        for key, values in metrics.items():
            # Convert numpy arrays to Python lists first
            if isinstance(values, np.ndarray):
                values = values.tolist()
            elif isinstance(values, list):
                # Convert any numpy scalars in the list
                values = [float(v) if isinstance(v, np.floating) else 
                         int(v) if isinstance(v, np.integer) else 
                         v for v in values]
            
            # Calculate stats using Python native types
            if values:
                stats[f'{key}_mean'] = float(np.mean(values))
                stats[f'{key}_std'] = float(np.std(values))
                stats[f'{key}_min'] = float(np.min(values))
                stats[f'{key}_max'] = float(np.max(values))
            else:
                stats[f'{key}_mean'] = 0.0
                stats[f'{key}_std'] = 0.0
                stats[f'{key}_min'] = 0.0
                stats[f'{key}_max'] = 0.0
        
        # Print results
        print("\n" + "-" * 40)
        print("EVALUATION RESULTS:")
        print("-" * 40)
        print(f"Average Reward: {stats['rewards_mean']:.2f} ± {stats['rewards_std']:.2f}")
        print(f"Average Forward Distance: {stats['forward_distances_mean']:.2f} ± {stats['forward_distances_std']:.2f} meters")
        print(f"Average Episode Length: {stats['episode_lengths_mean']:.2f} ± {stats['episode_lengths_std']:.2f} steps")
        print(f"Energy Efficiency: {stats['energy_efficiency_mean']:.4f} meters/step")
        print("-" * 40)
        
        # return stats, metrics
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert all stats to serializable format
        serializable_stats = convert_to_serializable(stats)
        
        results_file = self.config.results_dir / f"{algorithm}_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        return stats, metrics
    
    def compare_algorithms(self, algorithm_results):
        """Compare performance of multiple algorithms"""
        print(f"\n{'='*60}")
        print("COMPARING ALGORITHM PERFORMANCE")
        print(f"{'='*60}")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        comparison_data = []
        
        # Helper function for JSON serialization
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        for idx, (algorithm, results) in enumerate(algorithm_results.items()):
            stats, metrics = results
            
            # Convert stats to serializable format
            stats_serializable = convert_to_serializable(stats)
            
            # Add to comparison data
            comparison_data.append({
                'Algorithm': algorithm,
                'Average Reward': float(stats_serializable.get('rewards_mean', 0)),
                'Reward Std': float(stats_serializable.get('rewards_std', 0)),
                'Forward Distance': float(stats_serializable.get('forward_distances_mean', 0)),
                'Episode Length': float(stats_serializable.get('episode_lengths_mean', 0)),
                'Energy Efficiency': float(stats_serializable.get('energy_efficiency_mean', 0))
            })
            
            # Plot reward distribution
            if 'rewards' in metrics and metrics['rewards']:
                rewards_serializable = convert_to_serializable(metrics['rewards'])
                axes[0].hist(rewards_serializable, alpha=0.5, label=algorithm, bins=10)
            
            # Plot forward distance vs reward
            if 'forward_distances' in metrics and metrics['forward_distances'] and 'rewards' in metrics and metrics['rewards']:
                distances_serializable = convert_to_serializable(metrics['forward_distances'])
                rewards_serializable = convert_to_serializable(metrics['rewards'])
                axes[1].scatter(distances_serializable, rewards_serializable, 
                              label=algorithm, alpha=0.6)
        
        axes[0].set_title('Reward Distribution')
        axes[0].set_xlabel('Total Reward')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        axes[1].set_title('Forward Distance vs Reward')
        axes[1].set_xlabel('Forward Distance (m)')
        axes[1].set_ylabel('Total Reward')
        axes[1].legend()
        
        # Create bar plots for comparison
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            ax2 = axes[2]
            x = np.arange(len(df))
            width = 0.35
            
            ax2.bar(x - width/2, df['Average Reward'], width, label='Average Reward', 
                   yerr=df['Reward Std'], capsize=5)
            ax2.bar(x + width/2, df['Forward Distance'], width, label='Forward Distance (m)')
            ax2.set_xlabel('Algorithm')
            ax2.set_ylabel('Score')
            ax2.set_title('Algorithm Performance Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(df['Algorithm'])
            ax2.legend()
            
            # Energy efficiency comparison
            ax3 = axes[3]
            bars = ax3.bar(df['Algorithm'], df['Energy Efficiency'])
            ax3.set_title('Energy Efficiency Comparison')
            ax3.set_xlabel('Algorithm')
            ax3.set_ylabel('Meters per Step')
            ax3.set_ylim(0, max(df['Energy Efficiency']) * 1.2 if len(df['Energy Efficiency']) > 0 else 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_plot_path = self.config.results_dir / "algorithm_comparison.png"
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {comparison_plot_path}")
        
        # Display results table
        print("\n" + "-" * 80)
        print("PERFORMANCE COMPARISON TABLE:")
        print("-" * 80)
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
            
            # Save comparison table - ensure all data is serializable
            comparison_csv_path = self.config.results_dir / "algorithm_comparison.csv"
            df.to_csv(comparison_csv_path, index=False)
            print(f"\nComparison table saved to: {comparison_csv_path}")
            
            # Also save as JSON
            comparison_json_path = self.config.results_dir / "algorithm_comparison.json"
            df_dict = df.to_dict(orient='records')
            with open(comparison_json_path, 'w') as f:
                json.dump(convert_to_serializable(df_dict), f, indent=2)
            print(f"Comparison JSON saved to: {comparison_json_path}")
        else:
            print("No comparison data available")
        
        print("-" * 80)
        
        plt.show()
        
        return df if comparison_data else pd.DataFrame()
    
 
        
class ProjectInterface:
    """Main user interface for the project"""
    
    def __init__(self):
        self.config = Config()
        self.visualizer = HumanoidVisualizer(self.config)
        self.current_trainer = None
        self.trained_models = {}
        
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("HUMANOID-V5")
        print("="*60)
        print("1. Display Environment Information")
        print("2. Train a New Model")
        print("3. Load and Test a Trained Model")
        print("4. Visualize Model Performance")
        print("5. Evaluate Model Quantitatively")
        print("6. Compare Algorithms")
        print("7. Display Available Trained Models")
        print("8. Run Complete Experiment Pipeline")
        print("9. Generate Project Report")
        print("0. Exit")
        print("="*60)
    
    def run(self):
        """Main run loop"""
        while True:
            self.display_menu()
            
            try:
                choice = input("\nEnter your choice (0-9): ").strip()
                
                if choice == '0':
                    print("\nExiting program. Goodbye!")
                    break
                
                elif choice == '1':
                    self.display_environment_info()
                
                elif choice == '2':
                    self.train_model_menu()
                
                elif choice == '3':
                    self.test_model_menu()
                
                elif choice == '4':
                    self.visualize_model_menu()
                
                elif choice == '5':
                    self.evaluate_model_menu()
                
                elif choice == '6':
                    self.compare_algorithms_menu()
                
                elif choice == '7':
                    self.display_available_models()
                
                elif choice == '8':
                    self.run_complete_experiment()
                
                elif choice == '9':
                    self.generate_report()
                
                else:
                    print("Invalid choice! Please enter a number between 0 and 9.")
            
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                continue
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                import traceback
                traceback.print_exc()
    
    def display_environment_info(self):
        """Display information about the Humanoid-v5 environment"""
        HumanoidEnvironment.print_env_info()
    
    def train_model_menu(self):
        """Menu for training a model"""
        print("\n" + "-"*40)
        print("TRAIN A NEW MODEL")
        print("-"*40)
        
        algorithms = ['PPO', 'SAC']
        
        print("Available algorithms:")
        for i, algo in enumerate(algorithms, 1):
            print(f"{i}. {algo}")
        
        try:
            algo_choice = int(input("\nSelect algorithm (1-4): ").strip())
            if 1 <= algo_choice <= 4:
                algorithm = algorithms[algo_choice - 1]
                
                # Get training parameters
                print(f"\nSelected algorithm: {algorithm}")
                print(f"Default total timesteps: {self.config.total_timesteps}")
                
                change_params = input("Change training parameters? (y/n): ").lower().strip()
                if change_params == 'y':
                    try:
                        total_timesteps = int(input("Enter total timesteps: "))
                        self.config.total_timesteps = total_timesteps
                    except ValueError:
                        print("Invalid input. Using default value.")
                
                # Create trainer and train
                self.current_trainer = HumanoidTrainer(algorithm=algorithm, config=self.config)
                model = self.current_trainer.train()
                
                # Store reference
                self.trained_models[algorithm] = {
                    'trainer': self.current_trainer,
                    'model': model,
                    'path': self.current_trainer.algorithm_dir / f"{algorithm}_final.zip"
                }
                
                print(f"\nTraining completed! Model saved.")
                
            else:
                print("Invalid choice!")
        
        except ValueError:
            print("Invalid input!")
    
    def test_model_menu(self):
        """Menu for testing a trained model"""
        print("\n" + "-"*40)
        print("TEST A TRAINED MODEL")
        print("-"*40)
        
        # Find available models
        model_files = list(self.config.models_dir.glob("**/*.zip"))
        
        if not model_files:
            print("No trained models found!")
            return
        
        print("Available models:")
        for i, model_file in enumerate(model_files, 1):
            model_name = model_file.parent.name
            print(f"{i}. {model_name} - {model_file.name}")
        
        try:
            model_choice = int(input("\nSelect model (0 to cancel): ").strip())
            if model_choice == 0:
                return
            
            if 1 <= model_choice <= len(model_files):
                selected_file = model_files[model_choice - 1]
                
                # Determine algorithm from filename
                algorithm = None
                for algo in ['PPO', 'SAC']:
                    if algo in selected_file.parent.name:
                        algorithm = algo
                        break
                
                if not algorithm:
                    algorithm = input("Could not detect algorithm. Enter algorithm name: ").strip()
                
                # Create trainer and load model
                trainer = HumanoidTrainer(algorithm=algorithm, config=self.config)
                model = trainer.load_model(str(selected_file))
                
                # Run a test episode
                print("\nRunning test episode...")
                env = HumanoidEnvironment.make_env(render_mode='human')
                obs, _ = env.reset()
                total_reward = 0
                
                for step in range(200):  # Run for 200 steps max
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    
                    if terminated or truncated:
                        break
                
                env.close()
                print(f"Test episode completed!")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Steps: {step + 1}")
                
            else:
                print("Invalid choice!")
        
        except ValueError:
            print("Invalid input!")
    
    def visualize_model_menu(self):
        """Menu for visualizing a trained model"""
        print("\n" + "-"*40)
        print("VISUALIZE MODEL PERFORMANCE")
        print("-"*40)
        
        # Find available models
        model_files = list(self.config.models_dir.glob("**/*_final.zip"))
        
        if not model_files:
            print("No trained models found!")
            return
        
        print("Available models:")
        for i, model_file in enumerate(model_files, 1):
            model_name = model_file.parent.name
            print(f"{i}. {model_name}")
        
        try:
            model_choice = int(input("\nSelect model (0 to cancel): ").strip())
            if model_choice == 0:
                return
            
            if 1 <= model_choice <= len(model_files):
                selected_file = model_files[model_choice - 1]
                
                # Determine algorithm from filename
                algorithm = None
                for algo in ['PPO', 'SAC']:
                    if algo in selected_file.parent.name:
                        algorithm = algo
                        break
                
                if not algorithm:
                    algorithm = input("Could not detect algorithm. Enter algorithm name: ").strip()
                
                # Get visualization parameters
                num_episodes = int(input("Number of episodes to visualize (default: 2): ") or "2")
                save_video = input("Save videos? (y/n, default: y): ").lower().strip() != 'n'
                
                # Run visualization
                print(f"\nVisualizing {algorithm} model...")
                episodes_data = self.visualizer.visualize_trained_model(
                    model_path=str(selected_file),
                    algorithm=algorithm,
                    num_episodes=num_episodes,
                    save_video=save_video
                )
                
            else:
                print("Invalid choice!")
        
        except ValueError:
            print("Invalid input!")
    
    def evaluate_model_menu(self):
        """Menu for quantitative evaluation"""
        print("\n" + "-"*40)
        print("QUANTITATIVE MODEL EVALUATION")
        print("-"*40)
        
        # Find available models
        model_files = list(self.config.models_dir.glob("**/*_final.zip"))
        
        if not model_files:
            print("No trained models found!")
            return
        
        print("Available models:")
        for i, model_file in enumerate(model_files, 1):
            model_name = model_file.parent.name
            print(f"{i}. {model_name}")
        
        try:
            model_choice = int(input("\nSelect model (0 to cancel): ").strip())
            if model_choice == 0:
                return
            
            if 1 <= model_choice <= len(model_files):
                selected_file = model_files[model_choice - 1]
                
                # Determine algorithm from filename
                algorithm = None
                for algo in ['PPO', 'SAC']:
                    if algo in selected_file.parent.name:
                        algorithm = algo
                        break
                
                if not algorithm:
                    algorithm = input("Could not detect algorithm. Enter algorithm name: ").strip()
                
                # Get evaluation parameters
                num_episodes = int(input("Number of evaluation episodes (default: 10): ") or "10")
                
                # Run evaluation
                print(f"\nEvaluating {algorithm} model...")
                stats, metrics = self.visualizer.evaluate_model(
                    model_path=str(selected_file),
                    algorithm=algorithm,
                    num_episodes=num_episodes
                )
                
            else:
                print("Invalid choice!")
        
        except ValueError:
            print("Invalid input!")
    
    def compare_algorithms_menu(self):
        """Menu for comparing multiple algorithms"""
        print("\n" + "-"*40)
        print("COMPARE MULTIPLE ALGORITHMS")
        print("-"*40)
        
        algorithms = ['PPO', 'SAC']
        results = {}
        
        print("Select algorithms to compare (comma-separated, e.g., 1,3,4):")
        for i, algo in enumerate(algorithms, 1):
            print(f"{i}. {algo}")
        
        try:
            choices = input("\nEnter choices: ").strip()
            selected_indices = [int(c.strip()) for c in choices.split(',') if c.strip().isdigit()]
            
            selected_algorithms = []
            for idx in selected_indices:
                if 1 <= idx <= len(algorithms):
                    selected_algorithms.append(algorithms[idx - 1])
            
            if not selected_algorithms:
                print("No valid algorithms selected!")
                return
            
            print(f"\nSelected algorithms: {', '.join(selected_algorithms)}")
            
            # Find models for each algorithm
            for algorithm in selected_algorithms:
                # Look for the latest model of this algorithm
                model_dirs = list(self.config.models_dir.glob(f"{algorithm}_*"))
                if not model_dirs:
                    print(f"No trained model found for {algorithm}!")
                    continue
                
                # Get the most recent directory
                latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                model_file = latest_dir / f"{algorithm}_final.zip"
                
                if model_file.exists():
                    print(f"\nEvaluating {algorithm}...")
                    stats, metrics = self.visualizer.evaluate_model(
                        model_path=str(model_file),
                        algorithm=algorithm,
                        num_episodes=5 
                    )
                    results[algorithm] = (stats, metrics)
                else:
                    print(f"Model file not found for {algorithm} in {latest_dir}")
            
            if len(results) >= 2:
                # Compare the algorithms
                self.visualizer.compare_algorithms(results)
            else:
                print("Need at least 2 algorithms with results to compare!")
        
        except ValueError:
            print("Invalid input!")
    
    def display_available_models(self):
        """Display all available trained models"""
        print("\n" + "="*60)
        print("AVAILABLE TRAINED MODELS")
        print("="*60)
        
        model_files = list(self.config.models_dir.glob("**/*.zip"))
        
        if not model_files:
            print("No trained models found!")
            print(f"Models directory: {self.config.models_dir}")
            return
        
        print(f"Found {len(model_files)} trained models:\n")
        
        for i, model_file in enumerate(model_files, 1):
            model_path = model_file.relative_to(self.config.project_root)
            parent_dir = model_file.parent.name
            file_size = model_file.stat().st_size / (1024 * 1024)  # MB
            
            print(f"{i:3}. {parent_dir}")
            print(f"     File: {model_file.name}")
            print(f"     Path: {model_path}")
            print(f"     Size: {file_size:.2f} MB")
            print()
    
    def run_complete_experiment(self):
        """Run the complete experiment pipeline"""
        print("\n" + "="*60)
        print("COMPLETE EXPERIMENT PIPELINE")
        print("="*60)
        print("This will:")
        print("1. Train the algorithms (PPO, SAC)")
        print("2. Evaluate each algorithm")
        print("3. Visualize the best model")
        print("4. Generate comparison plots")
        print("="*60)
        
        confirm = input("\nThis will take several hours. Continue? (y/n): ").lower().strip()
        if confirm != 'y':
            print("Experiment cancelled.")
            return
        
        algorithms = ['PPO', 'SAC']
        results = {}
        
        print("\nStarting complete experiment pipeline...")
        print(f"Project directory: {self.config.project_root}")
        
        # Step 1: Train all algorithms
        print("\n" + "-"*40)
        print("STEP 1: TRAINING ALGORITHMS")
        print("-"*40)
        
        for algorithm in algorithms:
            print(f"\nTraining {algorithm}...")
            try:
                trainer = HumanoidTrainer(algorithm=algorithm, config=self.config)
                model = trainer.train()
                
                self.trained_models[algorithm] = {
                    'trainer': trainer,
                    'model': model,
                    'path': trainer.algorithm_dir / f"{algorithm}_final.zip"
                }
                
                print(f"{algorithm} training completed!")
                
            except Exception as e:
                print(f"Error training {algorithm}: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 2: Evaluate all trained models
        print("\n" + "-"*40)
        print("STEP 2: EVALUATING MODELS")
        print("-"*40)
        
        for algorithm in algorithms:
            if algorithm in self.trained_models:
                model_path = self.trained_models[algorithm]['path']
                print(f"\nEvaluating {algorithm}...")
                
                try:
                    stats, metrics = self.visualizer.evaluate_model(
                        model_path=str(model_path),
                        algorithm=algorithm,
                        num_episodes=5
                    )
                    results[algorithm] = (stats, metrics)
                    
                except Exception as e:
                    print(f"Error evaluating {algorithm}: {e}")
        
        # Step 3: Compare algorithms
        print("\n" + "-"*40)
        print("STEP 3: COMPARING ALGORITHMS")
        print("-"*40)
        
        if len(results) >= 2:
            comparison_df = self.visualizer.compare_algorithms(results)
            
            # Find best algorithm
            best_algo = comparison_df.loc[comparison_df['Average Reward'].idxmax(), 'Algorithm']
            print(f"\n{'='*60}")
            print(f"BEST PERFORMING ALGORITHM: {best_algo}")
            print(f"{'='*60}")
            
            # Step 4: Visualize best model
            print("\n" + "-"*40)
            print("STEP 4: VISUALIZING BEST MODEL")
            print("-"*40)
            
            if best_algo in self.trained_models:
                model_path = self.trained_models[best_algo]['path']
                print(f"Visualizing {best_algo} (best performing model)...")
                
                try:
                    self.visualizer.visualize_trained_model(
                        model_path=str(model_path),
                        algorithm=best_algo,
                        num_episodes=2,
                        save_video=True
                    )
                except Exception as e:
                    print(f"Error visualizing {best_algo}: {e}")
        
        print("\n" + "="*60)
        print("EXPERIMENT PIPELINE COMPLETED!")
        print("="*60)
        print(f"All results saved in: {self.config.project_root}")
        print("Check the following directories:")
        print(f"  - Models: {self.config.models_dir}")
        print(f"  - Results: {self.config.results_dir}")
        print(f"  - Videos: {self.config.videos_dir}")
        print(f"  - Logs: {self.config.logs_dir}")
        print("="*60)
    
    def generate_report(self):
        """Generate a comprehensive project report"""
        print("\n" + "="*60)
        print("GENERATING PROJECT REPORT")
        print("="*60)
        
        report_path = self.config.results_dir / f"project_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # Collect project information
        model_files = list(self.config.models_dir.glob("**/*_final.zip"))
        video_files = list(self.config.videos_dir.glob("*.mp4"))
        result_files = list(self.config.results_dir.glob("*.json"))
        
        with open(report_path, 'w') as f:
            f.write("# Humanoid-v5 Reinforcement Learning Project Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Project Overview\n")
            f.write("This project trains and evaluates reinforcement learning algorithms on the Humanoid-v5 environment.\n\n")
            
            f.write("## Directory Structure\n")
            f.write(f"- Project Root: `{self.config.project_root}`\n")
            f.write(f"- Trained Models: `{self.config.models_dir}` ({len(model_files)} models)\n")
            f.write(f"- Results: `{self.config.results_dir}` ({len(result_files)} files)\n")
            f.write(f"- Videos: `{self.config.videos_dir}` ({len(video_files)} videos)\n")
            f.write(f"- Logs: `{self.config.logs_dir}`\n\n")
            
            f.write("## Available Models\n")
            if model_files:
                f.write("| Algorithm | Model Path |\n")
                f.write("|-----------|------------|\n")
                for model_file in model_files:
                    algo = model_file.parent.name.split('_')[0]
                    f.write(f"| {algo} | `{model_file.relative_to(self.config.project_root)}` |\n")
            else:
                f.write("No trained models found.\n")
            f.write("\n")
            
            f.write("## Available Videos\n")
            if video_files:
                for video_file in video_files:
                    f.write(f"- `{video_file.name}`\n")
            else:
                f.write("No videos found.\n")
            f.write("\n")
            
            f.write("## Environment Information\n")
            f.write("- **Name**: Humanoid-v5\n")
            f.write("- **Observation Space**: 376 dimensions\n")
            f.write("- **Action Space**: 17 dimensions (continuous)\n")
            f.write("- **Reward Function**: Forward progress + alive bonus - control cost\n\n")
            
            f.write("## Algorithms Implemented\n")
            f.write("1. **PPO** (Proximal Policy Optimization)\n")
            f.write("2. **SAC** (Soft Actor-Critic)\n")
            
            f.write("## How to Use\n")
            f.write("### 1. Run the Interface\n")
            f.write("```bash\npython humanoid_rl_project.py\n```\n\n")
            
            f.write("### 2. Train a Model\n")
            f.write("Select option 2 from the menu and choose an algorithm.\n\n")
            
            f.write("### 3. Visualize Results\n")
            f.write("Use options 4-6 to visualize and compare trained models.\n\n")
            
            f.write("### 4. View TensorBoard Logs\n")
            f.write("```bash\ntensorboard --logdir=humanoid_rl_project/logs/\n```\n\n")
            
            f.write("## Notes\n")
            f.write("- Training requires significant computational resources\n")
            f.write("- Each algorithm trains for 500,000 timesteps by default\n")
            f.write("- Videos are saved in MP4 format for visualization\n")
            f.write("- All metrics are saved as JSON files for analysis\n")
        
        print(f"\nReport generated: {report_path}")
        print("\nTo view TensorBoard logs:")
        print(f"  tensorboard --logdir={self.config.logs_dir}")
        print("\nSummary of available data:")
        print(f"  - Trained models: {len(model_files)}")
        print(f"  - Result files: {len(result_files)}")
        print(f"  - Videos: {len(video_files)}")
        print("\n" + "="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def check_dependencies():
    """Check and install required dependencies"""
    print("Checking dependencies...")
    
    required_packages = [
        ('gymnasium[box2d]', 'gymnasium'),
        ('stable-baselines3[extra]', 'stable_baselines3'),
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pandas', 'pandas'),
        ('imageio', 'imageio'),
        ('opencv-python', 'cv2'),
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    for package, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        install = input("Install missing packages? (y/n): ").lower().strip()
        
        if install == 'y':
            import subprocess
            import sys
            
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
            print("\nAll packages installed successfully!")
            print("Please restart the program.")
            return False
        else:
            print("\nCannot run without required packages.")
            return False
    
    return True


def print_welcome():
    """Print welcome message"""
    print("\n" + "="*70)
    print(" " * 20 + "Artifical Intelligence Foundation Project")
    print("="*70)
    print("\nHUMANOID-V5 Using RL")
    print("\nHere you can:")
    print("  • Train RL algorithms (PPO, SAC) on Humanoid-v5")
    print("  • Visualize trained humanoids walking")
    print("  • Compare algorithm performance")
    print("  • Generate reports for your paper")
    print("\n" + "-"*70)
    print("IMPORTANT NOTES:")
    print("- Training requires significant computation (GPU recommended)")
    print("- Each algorithm trains for 500,000 timesteps by default")
    print("- Complete experiment may take several hours")
    print("- Results will be  saved in 'humanoid_rl_project/' directory")
    print("-"*70)


def main():
    """Main function to run the project interface"""
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Print welcome message
    print_welcome()
    
    # Initialize and run the interface
    try:
        interface = ProjectInterface()
        interface.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your installation and try again.")


if __name__ == "__main__":
    main()