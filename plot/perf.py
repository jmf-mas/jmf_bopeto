import matplotlib.pyplot as plt
import seaborn as sns

def get_plot(df_perf, metric, already = False):
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(12, 5.4), sharey=False, sharex=False)
    for k in [0, 1, 2, 3]:
        for i, (subplot, df_group) in enumerate(df_perf.groupby('dataset')):
            ax = axes[k, i]
            unique_contamination = df_group['true_contamination'].unique()
            cont = unique_contamination[k]
            data = df_group[df_group['true_contamination']==cont]
            sns.lineplot(data=data, x=data.contamination, y=metric, hue='model', ax=ax, marker='X', palette=palette_model)
            ax.axvline(x=cont, color='r', linestyle='--')
            ax.set_xlabel('')
            ax.set_ylabel('')
            if k==0:
                ax.set_title(f'{subplot}')
            if i == 0:
                if k==0 and not already:
                    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.025), ncol=5)
                    already = True
            ax.legend().remove()
    plt.tight_layout()
    plt.savefig("cont/all_contamination_"+metric+".eps")
    plt.show()
