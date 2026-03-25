"""
BiasSpectra CLI – Command-line entry point.
============================================
Usage:
    python run.py predict "Opposition criticizes govt on farm laws"
    python run.py predict --model baseline "headline text"
    python run.py train --model baseline
    python run.py train --model bert
    python run.py evaluate
    python run.py app
"""

import argparse
import logging
import sys
import os

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_predict(args):
    """Run a single prediction."""
    from src.inference.predictor import BiasPredictor

    predictor = BiasPredictor(model_type=args.model)
    result = predictor.predict(args.headline)

    print(f"\n{'─' * 50}")
    print(f"  Headline : {args.headline}")
    print(f"  Bias     : {result.label}")
    print(f"  Gate     : {result.gate}")
    print(f"  Reason   : {result.reasoning}")
    if result.is_model_prediction:
        print(f"  Confidence:")
        for label, conf in result.confidence.items():
            bar = "█" * int(conf * 30) + "░" * (30 - int(conf * 30))
            print(f"    {label:>7s}  {bar}  {conf:.1%}")
    print(f"{'─' * 50}\n")


def cmd_train(args):
    """Run model training."""
    if args.model == "baseline":
        from src.training.baseline import BaselineTrainer
        trainer = BaselineTrainer()
        trainer.train()
    elif args.model == "bert":
        from src.training.bert_trainer import BertTrainer
        trainer = BertTrainer()
        trainer.train()
    else:
        print(f"Unknown model type: {args.model}")
        sys.exit(1)


def cmd_evaluate(args):
    """Run model evaluation."""
    from src.evaluation.evaluator import ModelEvaluator

    evaluator = ModelEvaluator()

    if args.model in ("all", "baseline"):
        try:
            evaluator.evaluate_baseline()
        except FileNotFoundError:
            print("⚠️ Baseline model not found – skipping.")

    if args.model in ("all", "bert"):
        try:
            evaluator.evaluate_bert()
        except FileNotFoundError:
            print("⚠️ BERT model not found – skipping.")


def cmd_app(args):
    """Launch the Streamlit app."""
    import subprocess
    app_path = os.path.join(os.path.dirname(__file__), "src", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


def main():
    parser = argparse.ArgumentParser(
        prog="biasspectra",
        description="BiasSpectra – Political Bias Detection for Indian News",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # predict
    p_predict = subparsers.add_parser("predict", help="Predict bias for a headline")
    p_predict.add_argument("headline", type=str, help="News headline to analyze")
    p_predict.add_argument(
        "--model", choices=["nli", "bert", "baseline"], default="nli",
        help="Model to use (default: nli)",
    )
    p_predict.set_defaults(func=cmd_predict)

    # train
    p_train = subparsers.add_parser("train", help="Train a model")
    p_train.add_argument(
        "--model", choices=["baseline", "bert"], required=True,
        help="Model type to train",
    )
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model performance")
    p_eval.add_argument(
        "--model", choices=["all", "baseline", "bert"], default="all",
        help="Which legacy model to evaluate (default: all)",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    # app
    p_app = subparsers.add_parser("app", help="Launch Streamlit web app")
    p_app.set_defaults(func=cmd_app)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
