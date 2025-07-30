"""
Unified Training Launcher for ChemBERTa and Chemprop
Easy-to-use interface for launching training jobs with W&B integration
"""

import modal
import click
from typing import Optional

app = modal.App("training-launcher")

@app.local_entrypoint()
@click.command()
@click.option('--model', type=click.Choice(['chemberta', 'chemprop', 'both']), 
              default='both', help='Which model to train')
@click.option('--dataset', default='oncoprotein_multitask_dataset', 
              help='Dataset name (without .csv extension)')
@click.option('--epochs', default=10, help='Number of training epochs')
@click.option('--batch-size', default=16, help='Batch size for training')
@click.option('--learning-rate', default=2e-5, help='Learning rate')
@click.option('--run-name', default=None, help='Custom W&B run name')
@click.option('--test-size', default=0.2, help='Test set size (fraction)')
@click.option('--val-size', default=0.1, help='Validation set size (fraction)')
def launch_training(
    model: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    run_name: Optional[str],
    test_size: float,
    val_size: float
):
    """
    Launch ChemBERTa and/or Chemprop training with W&B logging
    
    Examples:
    modal run launch_training.py --model chemberta --epochs 20
    modal run launch_training.py --model chemprop --batch-size 32
    modal run launch_training.py --model both --run-name "comparison-run"
    """
    
    print("🚀 Launching Training Pipeline(s)")
    print(f"📊 Dataset: {dataset}")
    print(f"🎯 Model(s): {model}")
    print(f"⚙️ Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    
    results = {}
    
    if model in ['chemberta', 'both']:
        print("\n🤖 Starting ChemBERTa Training...")
        
        # Import and run ChemBERTa training
        from train_chemberta import train_chemberta_multitask
        
        chemberta_result = train_chemberta_multitask.remote(
            dataset_name=dataset,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            test_size=test_size,
            val_size=val_size,
            run_name=f"{run_name}-chemberta" if run_name else None
        )
        
        results['chemberta'] = chemberta_result
        print(f"✅ ChemBERTa Training Completed: {chemberta_result['status']}")
    
    if model in ['chemprop', 'both']:
        print("\n🧪 Starting Chemprop Training...")
        
        # Import and run Chemprop training
        from train_chemprop import train_chemprop_multitask
        
        # Adjust parameters for Chemprop (different optimal ranges)
        chemprop_lr = max(learning_rate * 10, 1e-4)  # Chemprop typically needs higher LR
        chemprop_batch = min(batch_size * 3, 100)    # Chemprop can handle larger batches
        chemprop_epochs = max(epochs * 3, 30)        # Chemprop typically needs more epochs
        
        chemprop_result = train_chemprop_multitask.remote(
            dataset_name=dataset,
            num_epochs=chemprop_epochs,
            batch_size=chemprop_batch,
            learning_rate=chemprop_lr,
            test_size=test_size,
            val_size=val_size,
            run_name=f"{run_name}-chemprop" if run_name else None
        )
        
        results['chemprop'] = chemprop_result
        print(f"✅ Chemprop Training Completed: {chemprop_result['status']}")
    
    print("\n🎉 All Training Completed!")
    print("📊 W&B Results available at: https://wandb.ai/")
    print("🔗 Project: veridica-ai-training")
    
    # Print summary
    for model_name, result in results.items():
        print(f"\n📈 {model_name.upper()} Results:")
        if 'test_results' in result:
            for metric, value in result['test_results'].items():
                print(f"  {metric}: {value:.4f}")
        if 'wandb_run_id' in result:
            print(f"  W&B Run ID: {result['wandb_run_id']}")
    
    return results

if __name__ == "__main__":
    launch_training()