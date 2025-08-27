#!/usr/bin/env python3
"""
Command-line interface for Shipboard Fire Response RL System
"""

import click
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shipboard_rl.training.trainer import TrainingManager
from shipboard_rl.scenario.generator import ShipboardScenarioGenerator


@click.group()
def cli():
    """Shipboard Fire Response RL System CLI"""
    pass


@cli.command()
@click.option("--episodes", default=1000, help="Number of training episodes")
@click.option("--save-interval", default=200, help="Save model every N episodes")
@click.option("--output-dir", default="./models", help="Directory to save models")
def train(episodes, save_interval, output_dir):
    """Train the Enhanced DQN model"""
    click.echo("🔥 Starting Shipboard Fire Response RL Training")
    
    trainer = TrainingManager(save_dir=output_dir)
    report = trainer.train(num_episodes=episodes, save_interval=save_interval)
    
    click.echo(f"\n✅ Training completed!")
    click.echo(f"📊 Final average reward: {report['training_summary']['final_average_reward']:.2f}")
    
    # Save plots
    plots_path = Path(output_dir) / "training_plots.png"
    trainer.plot_training_progress(str(plots_path))


@cli.command()
@click.option("--model-path", required=True, help="Path to trained model")
@click.option("--episodes", default=100, help="Number of evaluation episodes")
def evaluate(model_path, episodes):
    """Evaluate a trained model"""
    click.echo("🧪 Evaluating Shipboard Fire Response RL Model")
    
    trainer = TrainingManager()
    trainer.agent.load_model(model_path)
    
    report = trainer.evaluate_model(num_episodes=episodes)
    
    click.echo(f"📊 Average reward: {report['average_reward']:.2f} ± {report['std_reward']:.2f}")
    click.echo(f"📈 Range: [{report['min_reward']:.2f}, {report['max_reward']:.2f}]")


@cli.command()
@click.option("--count", default=100, help="Number of scenarios to generate")
@click.option("--output", default="scenarios.json", help="Output file")
@click.option("--difficulty", type=click.Choice(['easy', 'medium', 'hard', 'extreme', 'mixed']), 
              default='mixed', help="Scenario difficulty")
def generate_scenarios(count, output, difficulty):
    """Generate training scenarios"""
    click.echo("🎲 Generating Shipboard Fire Response Scenarios")
    
    generator = ShipboardScenarioGenerator()
    
    if difficulty == 'mixed':
        scenarios = generator.generate_scenario_batch(count, difficulty_mix=True)
    else:
        scenarios = [generator.generate_scenario(difficulty) for _ in range(count)]
    
    # Save scenarios
    from shipboard_rl.scenario.generator import save_scenarios
    save_scenarios(scenarios, output)
    
    click.echo(f"✅ Generated {len(scenarios)} scenarios saved to {output}")


@cli.command()
def demo():
    """Run a quick demo of the system"""
    click.echo("🚢 Shipboard Fire Response RL System Demo")
    click.echo("=" * 50)
    
    # Generate sample scenarios
    generator = ShipboardScenarioGenerator(seed=42)
    scenarios = [
        generator.generate_scenario("easy"),
        generator.generate_scenario("medium"), 
        generator.generate_scenario("hard")
    ]
    
    for i, scenario in enumerate(scenarios):
        click.echo(f"\n📋 Sample Scenario {i+1}:")
        click.echo(f"   Compartment: {scenario.compartment.value}")
        click.echo(f"   Fire Type: {scenario.fire_type.value}")
        click.echo(f"   Condition: {scenario.ship_condition.value}")
        click.echo(f"   Intensity: {scenario.initial_intensity:.2f}")
        click.echo(f"   Crew Experience: {scenario.crew_experience:.2f}")
    
    click.echo(f"\n🎯 System ready for training!")
    click.echo(f"   Use 'python cli.py train' to start training")
    click.echo(f"   Use 'python cli.py generate-scenarios' to create training data")


if __name__ == "__main__":
    cli()
