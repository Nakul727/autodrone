"""
post_train_graphs.py
Generate training graphs showing key learning metrics
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

def create_reward_graph(episode_data, metrics_data, graphs_dir, model_name):
    """
    Total reward progress over time
    """
    plt.figure(figsize=(12, 8))

    # Plot individual episode rewards as faint dots
    if len(episode_data) > 0:
        plt.scatter(
            episode_data['episode'], episode_data['total_reward'],
            color='blue', alpha=0.15, s=8, label='Individual Episodes'
        )

    # Plot rolling mean
    if len(metrics_data) > 0:
        plt.plot(metrics_data['episode'], metrics_data['total_reward_mean'], color='blue', linewidth=3, label='Rolling Mean (100 episodes)')
        upper_bound = metrics_data['total_reward_mean'] + metrics_data['total_reward_std']
        lower_bound = metrics_data['total_reward_mean'] - metrics_data['total_reward_std']
        plt.fill_between(metrics_data['episode'], lower_bound, upper_bound, alpha=0.15, color='gray', label='±1 Standard Deviation')

    # Save graph
    plt.title('Total Reward Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'{model_name}_reward_progress.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_distance_graph(episode_data, metrics_data, graphs_dir, model_name):
    """
    Distance to target progress - emphasizing successful episodes
    """
    plt.figure(figsize=(12, 8))
    
    success_threshold = 0.15

    # Add success threshold line
    plt.axhline(y=success_threshold, color='red', linestyle='--', linewidth=3, alpha=0.5, label=f'Success Threshold ({success_threshold}m)')

    # Plot individual episode distances: green if succeeded, red if not
    success_col = 'at_target'
    if len(episode_data) > 0:
        colors = episode_data[success_col].map({True: 'green', False: 'red'})
        alphas = episode_data[success_col].map({True: 0.25, False: 0.12})
        sizes = episode_data[success_col].map({True: 12, False: 8})
        for i in range(len(episode_data)):
            plt.scatter(
                episode_data['episode'].iloc[i],
                episode_data['final_distance'].iloc[i],
                color=colors.iloc[i],
                alpha=alphas.iloc[i],
                s=sizes.iloc[i]
            )
    
    # Plot rolling mean
    if len(metrics_data) > 0:
        plt.plot(metrics_data['episode'], metrics_data['final_distance_mean'], color='orange', linewidth=2)
        upper = metrics_data['final_distance_mean'] + metrics_data['final_distance_std']
        lower = np.maximum(metrics_data['final_distance_mean'] - metrics_data['final_distance_std'], 0)
        plt.fill_between(metrics_data['episode'], lower, upper, alpha=0.1, color='gray')

    # Save graph
    plt.title('Distance to Target Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Final Distance to Target (m)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'{model_name}_distance_progress.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_episode_length_graph(episode_data, metrics_data, graphs_dir, model_name):
    """
    Episode length progress over time.
    """
    plt.figure(figsize=(12, 8))
 

    if len(metrics_data) > 0:
        plt.plot(
            metrics_data['episode'],
            metrics_data['episode_length_mean'],
            color='purple',
            linewidth=3,
            label='Rolling Mean Episode Length'
        )
        upper = metrics_data['episode_length_mean'] + metrics_data['episode_length_std']
        lower = np.maximum(metrics_data['episode_length_mean'] - metrics_data['episode_length_std'], 0)
        plt.fill_between(
            metrics_data['episode'],
            lower,
            upper,
            alpha=0.15,
            color='gray',
            label='±1 Standard Deviation'
        )
    
    # Save graph
    plt.title('Episode Length Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Length (steps)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'{model_name}_episode_length_progress.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_success_rate_graph(episode_data, metrics_data, graphs_dir, model_name):
    """
    Success rate progress over time.
    """
    # Graph config
    plt.figure(figsize=(12, 8))

    if len(metrics_data) > 0:
        plt.plot(
            metrics_data['episode'],
            metrics_data['success_rate_mean'] * 100,
            color='green',
            linewidth=3,
            label='Rolling Mean Success Rate'
        )
        upper = np.minimum(metrics_data['success_rate_mean'] * 100 + metrics_data['success_rate_std'] * 100, 100)
        lower = np.maximum(metrics_data['success_rate_mean'] * 100 - metrics_data['success_rate_std'] * 100, 0)
        plt.fill_between(
            metrics_data['episode'],
            lower,
            upper,
            alpha=0.15,
            color='gray',
            label='±1 Standard Deviation'
        )
    
    # Save graph
    plt.title('Success Rate Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend(fontsize=11)
    plt.savefig(os.path.join(graphs_dir, f'{model_name}_success_rate_progress.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def calculate_rolling_metrics(episode_data, window_size=100):
    """
    Calculate rolling metrics if metrics file doesn't exist
    """
    metrics = []
    success_col = 'at_target'
    for i in range(window_size, len(episode_data) + 1):
        window = episode_data.iloc[i-window_size:i]
        
        # Calculate standard deviations
        episode_length_std = window['episode_length'].std() if len(window) > 1 else 0.0
        success_rate_std = window[success_col].std() if len(window) > 1 else 0.0
        final_distance_std = window['final_distance'].std() if len(window) > 1 else 0.0
        total_reward_std = window['total_reward'].std() if len(window) > 1 else 0.0
        
        metrics.append({
            'episode': i - 1,
            'success_rate_mean': window[success_col].mean(),
            'success_rate_std': success_rate_std,
            'final_distance_mean': window['final_distance'].mean(),
            'final_distance_std': final_distance_std,
            'total_reward_mean': window['total_reward'].mean(),
            'total_reward_std': total_reward_std,
            'episode_length_mean': window['episode_length'].mean(),
            'episode_length_std': episode_length_std,
        })

    return pd.DataFrame(metrics)

def create_graphs(log_dir, graphs_dir, model_name):
    """
    Create training graphs from episode and metric logs
    """
    episode_file = os.path.join(log_dir, "episode_log.csv")
    metrics_file = os.path.join(log_dir, "metrics_log.csv")

    # Load episode data
    if not os.path.exists(episode_file):
        print("No episode log file found")
        return graphs_dir
    episode_data = pd.read_csv(episode_file)
    
    # Load metrics if available, otherwise calculate 
    if os.path.exists(metrics_file):
        metrics_data = pd.read_csv(metrics_file)
    else:
        metrics_data = calculate_rolling_metrics(episode_data)
    
    # Generate graphs
    create_reward_graph(episode_data, metrics_data, graphs_dir, model_name)
    create_distance_graph(episode_data, metrics_data, graphs_dir, model_name)
    create_episode_length_graph(episode_data, metrics_data, graphs_dir, model_name)
    create_success_rate_graph(episode_data, metrics_data, graphs_dir, model_name)
    return graphs_dir

def generate_all_graphs(log_dir, graphs_dir, model_name):
    """
    Generate all focused training graphs
    """
    graphs_dir = create_graphs(log_dir, graphs_dir, model_name)
    return graphs_dir