import sys
import argparse
from stable_baselines3 import PPO
from helper import create_env, parse_args, prompt_args
from eval_seq import evaluate_sequence
from eval_single import evaluate_single

def main():
    """
    Main entry point for evaluation.
    Determines whether to use CLI args or prompt-based input,
    loads the trained model, and runs evaluation based on the task.
    """
    args = parse_args() if len(sys.argv) > 1 else prompt_args()

    model = PPO.load(args.model_path)

    env = create_env(gui=args.gui, record=args.record)

    if args.task == "evaluate_single":
        evaluate_single(model, env, args.episodes, args.speed, start_pos=args.start_pos)
    else:
        evaluate_sequence(model, env, num_targets=args.targets, speed=args.speed, start_pos=args.start_pos)

    env.close()

if __name__ == "__main__":
    main()

