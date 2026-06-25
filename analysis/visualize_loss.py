import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Sample data (replace this with your actual data file)
data_string = """0,0,5.1219377517700195,0.9940355459415499,121.0,3.0,"1.0::;(;(;(;(;(target(X);type(X;~man));expression(~wearing;X;~shirt));expression(~on;~shirt;X));expression(~in;X;~shirt));expression(~contains;X;~shirt)).","('man'; 'bbox3')",0.0,0.0
0,1,18.50625991821289,0.9999999908201954,143.0,3.0,"1.0::;(;(;(;(;(target(X);type(X;~man));expression(~wearing;X;~shirt));expression(~on;~shirt;X));expression(~in;X;~shirt));expression(~contains;X;~shirt)).","('bike'; 'bbox6')",0.0,0.0
0,2,0.0,0.0,24.0,3.0,"1.0::;(;(target(X);type(X;~backpack));expression(~wears;~man;X)).","('backpack'; 'bbox2')",0.0,0.0
0,3,20.0,0.9999999979388464,6.0,0.0,"1.0::;(;(target(X);type(X;~car));expression(~has;X;~snowy)).","('car'; None)",0.0,0.0
0,4,0.0,0.0,26.0,3.0,"1.0::;(;(target(X);type(X;~girl));expression(~wears;X;~chain)).","('girl'; 'bbox1')",0.0,0.0
0,5,0.0,0.0,21.0,3.0,"1.0::;(;(target(X);type(X;~pen));expression(~on;X;~desk)).","('pen'; 'bbox4')",0.0,0.0
0,6,0.0,0.0,26.0,3.0,"1.0::;(;(target(X);type(X;~chain));expression(~wears;~girl;X)).","('chain'; 'bbox3')",0.0,0.0"""

def load_and_visualize_loss(data_source=None, file_path=None):
    """
    Load training data and visualize loss over epochs for each instance.
    
    Parameters:
    data_source: string containing CSV data (for testing)
    file_path: path to CSV file containing the data
    """
    
    # Column names based on the data structure
    columns = ['epoch', 'instance_id', 'loss', 'confidence', 'col4', 'col5', 
               'expression', 'object_info', 'col8', 'col9']
    
    # Load data
    if data_source:
        df = pd.read_csv(file_path, header=None, names=columns, skiprows=1)

    elif file_path:
        df = pd.read_csv(file_path, header=None, names=columns, skiprows=1)

    else:
        raise ValueError("Either data_source or file_path must be provided")
    
    # Clean and prepare data
    df['epoch'] = df['epoch'].astype(int)
    df['instance_id'] = df['instance_id'].astype(int)
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
    
    # Extract object type from object_info for better labeling
    df['object_type'] = df['object_info'].str.extract(r"'([^']+)'")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Line plot showing loss over epochs for each instance
    plt.subplot(2, 2, 1)
    for instance_id in df['instance_id'].unique():
        instance_data = df[df['instance_id'] == instance_id].sort_values('epoch')
        object_type = instance_data['object_type'].iloc[0]
        plt.plot(instance_data['epoch'], instance_data['loss'], marker = 'o') 
                # marker='o', label=f'Instance {instance_id} ({object_type})')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs by Instance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap of loss values
    plt.subplot(2, 2, 2)
    pivot_data = df.pivot(index='instance_id', columns='epoch', values='loss')
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Loss Heatmap (Instance vs Epoch)')
    plt.ylabel('Instance ID')
    plt.xlabel('Epoch')
    
    # Plot 3: Bar plot of final epoch loss by instance
    plt.subplot(2, 2, 3)
    final_epoch = df['epoch'].max()
    final_losses = df[df['epoch'] == final_epoch]
    bars = plt.bar(final_losses['instance_id'], final_losses['loss'])
    plt.xlabel('Instance ID')
    plt.ylabel('Loss')
    plt.title(f'Final Loss by Instance (Epoch {final_epoch})')
    plt.xticks(final_losses['instance_id'])
    
    # Color bars by loss value
    for bar, loss in zip(bars, final_losses['loss']):
        if loss > 10:
            bar.set_color('red')
        elif loss > 5:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    # Plot 4: Loss distribution
    plt.subplot(2, 2, 4)
    plt.hist(df['loss'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Loss Values')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== Loss Summary Statistics ===")
    summary_stats = df.groupby('instance_id')['loss'].agg(['mean', 'std', 'min', 'max'])
    summary_stats['object_type'] = df.groupby('instance_id')['object_type'].first()
    print(summary_stats)
    
    return df

# Example usage:
if __name__ == "__main__":

    
    # Option 2: Use with your own CSV file
    df = load_and_visualize_loss(file_path='training_metrics.csv')
    
    # Additional analysis
    print(f"\nTotal epochs: {df['epoch'].max() - df['epoch'].min() + 1}")
    print(f"Number of instances: {df['instance_id'].nunique()}")
    print(f"Average loss across all instances: {df['loss'].mean():.4f}")
    
    # Identify problematic instances (high loss)
    high_loss_threshold = df['loss'].quantile(0.75)
    problematic_instances = df[df['loss'] > high_loss_threshold]
    if not problematic_instances.empty:
        print(f"\nInstances with high loss (>{high_loss_threshold:.2f}):")
        print(problematic_instances[['instance_id', 'object_type', 'loss']].to_string(index=False))