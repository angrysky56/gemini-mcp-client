#!/usr/bin/env python3
"""
ADK Agent Model Configuration Utility
Allows changing the default model for all ADK agents.
"""

import argparse
import json
from pathlib import Path


def load_config():
    """Load the current agent configuration."""
    config_path = Path(__file__).parent / "config" / "models.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    else:
        return {
            "default_model": "gemini-2.0-flash-lite-001",
            "available_models": [
                "gemini-2.0-flash-lite-001",
                "gemini-2.5-flash-preview-05-20",
                "gemini-2.0-flash-preview-image-generation",
                "gemini-2.5-pro-preview-05-06"
                "gemma-3n-e4b-it"
            ],
            "model_descriptions": {
                "gemini-2.0-flash-lite": "30 rpm 1500 req/day (free)",
                "gemini-2.5-flash-preview-05-20": "10 rpm 500 req/day (free)",
                "gemini-2.0-flash-preview-image-generation": "15 rpm 1500 req/day (free) fast image model",
                "gemini-2.5-pro-preview-05-06": "More powerful model for complex tasks 5 rpm 25 req/day (free)",
                "gemma-3n-e4b-it": "Lightweight model 30 rpm 14400 req/day (free)"
            }
        }

def save_config(config):
    """Save the agent configuration."""
    config_path = Path(__file__).parent / "config" / "models.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def list_models():
    """List available models."""
    config = load_config()
    print("Available Models:")
    print("=" * 50)

    current_model = config["default_model"]

    for i, model in enumerate(config["available_models"], 1):
        status = "→ CURRENT" if model == current_model else ""
        description = config["model_descriptions"].get(model, "No description")
        print(f"{i:2d}. {model} {status}")
        print(f"    {description}")
        print()

def set_model(model_name):
    """Set the default model."""
    config = load_config()

    if model_name not in config["available_models"]:
        print(f"❌ Model '{model_name}' not available.")
        print("Available models:")
        for model in config["available_models"]:
            print(f"  - {model}")
        return False

    config["default_model"] = model_name
    save_config(config)
    print(f"✅ Default model set to: {model_name}")
    print("Restart ADK web interface to apply changes.")
    return True

def add_model(model_name, description=None):
    """Add a new model to the configuration."""
    config = load_config()

    if model_name in config["available_models"]:
        print(f"⚠️  Model '{model_name}' already exists.")
        return False

    config["available_models"].append(model_name)
    if description:
        config["model_descriptions"][model_name] = description

    save_config(config)
    print(f"✅ Added model: {model_name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="ADK Agent Model Configuration")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List models
    subparsers.add_parser("list", help="List available models")

    # Set model
    set_parser = subparsers.add_parser("set", help="Set default model")
    set_parser.add_argument("model", help="Model name to set as default")

    # Add model
    add_parser = subparsers.add_parser("add", help="Add new model")
    add_parser.add_argument("model", help="Model name to add")
    add_parser.add_argument("--description", help="Model description")

    args = parser.parse_args()

    if args.command == "list":
        list_models()
    elif args.command == "set":
        set_model(args.model)
    elif args.command == "add":
        add_model(args.model, args.description)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
