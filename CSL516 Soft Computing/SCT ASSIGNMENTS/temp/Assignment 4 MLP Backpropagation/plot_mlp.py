import sys
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv('mlp_metrics.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Subplot 1: Table of E_average values
    ax1.axis('tight'); ax1.axis('off')
    table_data = df.tail(15).round(5)
    table = ax1.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
    table.scale(1, 1.4)
    ax1.set_title('MLP Backpropagation: E_average (Last Epochs)', fontweight='bold')
    # Subplot 2: Non-linear Loss curve
    ax2.plot(df['Epoch'], df['E_average'], color='#6a1b9a', linewidth=2.5, label='Backprop MSE Loss')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('E_average (Mean Squared Error)', fontsize=12)
    ax2.set_title('MLP: Backpropagation Loss Reduction Curve', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    plt.tight_layout()
    plt.savefig('mlp_convergence_plot.png', dpi=150)
    print('[Python] Saved plot to mlp_convergence_plot.png')
    plt.show()
except ImportError:
    print('[Notice] Matplotlib/pandas not installed. Run: pip3 install pandas matplotlib')
