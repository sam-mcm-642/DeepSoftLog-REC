import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_and_combine_data():
    """Load both CSV files and combine them with adjusted epoch numbers."""
    
    # Load first CSV (epochs 0-3)
    print("Loading training_metrics.csv...")
    df1 = pd.read_csv('training_metrics.csv')
    print(f"First CSV: {len(df1)} rows, epochs {df1['epoch'].min()}-{df1['epoch'].max()}")
    
    # Load second CSV (epochs 0-34, need to adjust to 4-38)
    print("Loading training_metrics_next.csv...")
    df2 = pd.read_csv('training_metrics_next.csv')
    print(f"Second CSV: {len(df2)} rows, epochs {df2['epoch'].min()}-{df2['epoch'].max()}")
    
    # Adjust epochs in second dataset to continue from first
    max_epoch_first = df1['epoch'].max()
    df2['epoch'] = df2['epoch'] + max_epoch_first + 1
    print(f"Adjusted second CSV epochs to {df2['epoch'].min()}-{df2['epoch'].max()}")
    
    # Combine datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} rows, epochs {combined_df['epoch'].min()}-{combined_df['epoch'].max()}")
    
    return combined_df

def convert_loss_to_prob(df):
    """Convert loss from log prob form to regular probability."""
    # Assuming loss is negative log probability, convert to probability
    df['loss_prob'] = np.exp(-df['loss'])
    return df

def create_plots_directory():
    """Create train/plots directory if it doesn't exist."""
    plots_dir = Path('train/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir

def generate_loss_curves(df, plots_dir):
    """Generate individual loss and diff curves by epoch."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # Calculate epoch statistics
    epoch_stats = df.groupby('epoch').agg({
        'diff': ['mean', 'std', 'min', 'max'],
        'loss': ['mean', 'std'],
        'proof_steps': 'mean'
    }).reset_index()
    
    # Flatten column names
    epoch_stats.columns = ['_'.join(col).strip('_') for col in epoch_stats.columns]
    
    # 1. Diff curve
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_stats['epoch'], epoch_stats['diff_mean'], 'b-', linewidth=3, label='Mean Diff', marker='o', markersize=6)
    plt.fill_between(epoch_stats['epoch'], 
                    epoch_stats['diff_mean'] - epoch_stats['diff_std'],
                    epoch_stats['diff_mean'] + epoch_stats['diff_std'],
                    alpha=0.3, color='blue', label='±1 Std Dev')
    plt.title('Diff Values by Epoch', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Diff', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'diff_by_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_stats['epoch'], epoch_stats['loss_mean'], 'r-', linewidth=3, label='Mean Loss', marker='s', markersize=6)
    plt.fill_between(epoch_stats['epoch'], 
                    epoch_stats['loss_mean'] - epoch_stats['loss_std'],
                    epoch_stats['loss_mean'] + epoch_stats['loss_std'],
                    alpha=0.3, color='red', label='±1 Std Dev')
    plt.title('Loss by Epoch (Log Probability)', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (log prob)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'loss_by_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Proof steps
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_stats['epoch'], epoch_stats['proof_steps_mean'], 'g-', linewidth=3, marker='^', markersize=8)
    plt.title('Average Proof Steps by Epoch', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Proof Steps', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'proof_steps_by_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Loss distribution
    plt.figure(figsize=(10, 6))
    if len(df['epoch'].unique()) <= 20:  # Only if not too many epochs
        sns.violinplot(data=df, x='epoch', y='loss')
        plt.title('Loss Distribution by Epoch', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
    else:
        # For many epochs, show histogram of all losses
        plt.hist(df['loss'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Overall Loss Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Loss', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / 'loss_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: diff_by_epoch.png, loss_by_epoch.png, proof_steps_by_epoch.png, loss_distribution.png")
    
    return epoch_stats

def generate_additional_plots(df, plots_dir):
    """Generate individual additional analysis plots."""
    
    # 1. Proof steps distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['proof_steps'], bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title('Distribution of Proof Steps', fontsize=16, fontweight='bold')
    plt.xlabel('Proof Steps', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'proof_steps_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Loss vs Proof Steps scatter
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['proof_steps'], df['loss'], alpha=0.6, c=df['epoch'], cmap='viridis', s=20)
    plt.title('Loss vs Proof Steps (colored by epoch)', fontsize=16, fontweight='bold')
    plt.xlabel('Proof Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Epoch', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'loss_vs_proof_steps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Diff vs Loss scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(df['loss'], df['diff'], alpha=0.6, color='orange', s=20)
    plt.title('Diff vs Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Loss', fontsize=14)
    plt.ylabel('Diff', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'diff_vs_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Failed proofs by epoch
    failed_by_epoch = df[df['loss'] == 20.0].groupby('epoch').size()
    all_epochs = range(df['epoch'].min(), df['epoch'].max() + 1)
    failed_counts = [failed_by_epoch.get(epoch, 0) for epoch in all_epochs]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(all_epochs, failed_counts, alpha=0.7, color='red', edgecolor='darkred')
    plt.title('Failed Proofs by Epoch (loss = 20)', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Number of Failed Proofs', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, failed_counts):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'failed_proofs_by_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Batch-level loss progression (if not too many batches)
    if len(df) <= 5000:  # Only if dataset isn't too large
        plt.figure(figsize=(15, 6))
        for epoch in sorted(df['epoch'].unique()):
            epoch_data = df[df['epoch'] == epoch].sort_values('batch')
            plt.plot(epoch_data.index, epoch_data['loss'], alpha=0.7, label=f'Epoch {epoch}')
        
        plt.title('Batch-level Loss Progression', fontsize=16, fontweight='bold')
        plt.xlabel('Global Batch Index', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        if len(df['epoch'].unique()) <= 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'batch_level_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr_matrix = df[['loss', 'diff', 'proof_steps', 'epoch', 'batch']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Training Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: proof_steps_distribution.png, loss_vs_proof_steps.png, diff_vs_loss.png")
    print(f"✅ Saved: failed_proofs_by_epoch.png, correlation_heatmap.png")
    if len(df) <= 5000:
        print(f"✅ Saved: batch_level_loss.png")

def calculate_summary_statistics(df):
    """Calculate and print comprehensive summary statistics."""
    
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY STATISTICS")
    print("="*60)
    
    # Basic dataset info
    print(f"\nDataset Overview:")
    print(f"- Total samples: {len(df):,}")
    print(f"- Epochs: {df['epoch'].min()} to {df['epoch'].max()} ({df['epoch'].nunique()} total)")
    print(f"- Batches per epoch: {df.groupby('epoch')['batch'].max().mean():.1f} average")
    
    # Loss statistics
    print(f"\nLoss Statistics (log prob form):")
    print(f"- Mean: {df['loss'].mean():.4f}")
    print(f"- Std: {df['loss'].std():.4f}")
    print(f"- Min: {df['loss'].min():.4f}")
    print(f"- Max: {df['loss'].max():.4f}")
    print(f"- Median: {df['loss'].median():.4f}")
    
    # Failed proofs
    failed_proofs = (df['loss'] == 20.0).sum()
    print(f"\nFailed Proofs (loss = 20):")
    print(f"- Count: {failed_proofs:,}")
    print(f"- Percentage: {failed_proofs/len(df)*100:.2f}%")
    
    # Proof steps statistics
    print(f"\nProof Steps Statistics:")
    print(f"- Mean: {df['proof_steps'].mean():.2f}")
    print(f"- Std: {df['proof_steps'].std():.2f}")
    print(f"- Min: {df['proof_steps'].min():.0f}")
    print(f"- Max: {df['proof_steps'].max():.0f}")
    print(f"- Median: {df['proof_steps'].median():.0f}")
    
    # Diff statistics
    print(f"\nDiff Statistics:")
    print(f"- Mean: {df['diff'].mean():.6f}")
    print(f"- Std: {df['diff'].std():.6f}")
    print(f"- Min: {df['diff'].min():.6f}")
    print(f"- Max: {df['diff'].max():.6f}")
    
    # Per-epoch statistics
    print(f"\nPer-Epoch Statistics:")
    epoch_stats = df.groupby('epoch').agg({
        'loss': ['mean', 'std', 'min', 'max'],
        'diff': ['mean', 'std'],
        'proof_steps': ['mean', 'std'],
        'batch': 'count'
    })
    
    print(f"- Average loss per epoch: {epoch_stats[('loss', 'mean')].mean():.4f}")
    print(f"- Loss improvement (first to last epoch): {epoch_stats[('loss', 'mean')].iloc[0] - epoch_stats[('loss', 'mean')].iloc[-1]:.4f}")
    print(f"- Average diff per epoch: {epoch_stats[('diff', 'mean')].mean():.6f}")
    print(f"- Average proof steps per epoch: {epoch_stats[('proof_steps', 'mean')].mean():.2f}")
    
    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    correlations = df[['loss', 'diff', 'proof_steps', 'epoch']].corr()
    print(f"- Loss vs Proof Steps: {correlations.loc['loss', 'proof_steps']:.4f}")
    print(f"- Loss vs Diff: {correlations.loc['loss', 'diff']:.4f}")
    print(f"- Loss vs Epoch: {correlations.loc['loss', 'epoch']:.4f}")
    
    return epoch_stats

def save_combined_csv(df, filename='combined_training_metrics.csv'):
    """Save the combined dataset to CSV."""
    df.to_csv(filename, index=False)
    print(f"\nCombined dataset saved to: {filename}")
    print(f"File size: {os.path.getsize(filename) / 1024:.1f} KB")

def main():
    """Main analysis pipeline."""
    try:
        # Load and combine data
        df = load_and_combine_data()
        
        # Convert loss to probability form
        df = convert_loss_to_prob(df)
        
        # Create plots directory
        plots_dir = create_plots_directory()
        print(f"Plots will be saved to: {plots_dir}")
        
        # Generate plots
        print("\nGenerating individual plots...")
        epoch_stats = generate_loss_curves(df, plots_dir)
        generate_additional_plots(df, plots_dir)
        
        # Calculate and print statistics
        calculate_summary_statistics(df)
        
        # Save combined CSV
        save_combined_csv(df)
        
        # Print epoch-by-epoch summary
        print(f"\nEpoch-by-Epoch Summary:")
        print("-" * 80)
        for epoch in sorted(df['epoch'].unique()):
            epoch_data = df[df['epoch'] == epoch]
            failed_count = (epoch_data['loss'] == 20.0).sum()
            print(f"Epoch {epoch:2d}: {len(epoch_data):3d} batches, "
                  f"avg_loss={epoch_data['loss'].mean():6.3f}, "
                  f"avg_diff={epoch_data['diff'].mean():.6f}, "
                  f"avg_steps={epoch_data['proof_steps'].mean():5.1f}, "
                  f"failed={failed_count:2d}")
        
        print(f"\n✅ Analysis complete! Individual plots saved to train/plots/:")
        print(f"   - diff_by_epoch.png")
        print(f"   - loss_by_epoch.png") 
        print(f"   - proof_steps_by_epoch.png")
        print(f"   - loss_distribution.png")
        print(f"   - proof_steps_distribution.png")
        print(f"   - loss_vs_proof_steps.png")
        print(f"   - diff_vs_loss.png")
        print(f"   - failed_proofs_by_epoch.png")
        print(f"   - correlation_heatmap.png")
        if len(df) <= 5000:
            print(f"   - batch_level_loss.png")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find CSV file. {e}")
        print("Please ensure 'training_metrics.csv' and 'training_metrics_next.csv' are in the current directory.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()