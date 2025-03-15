import matplotlib.pyplot as plt
import seaborn as sns

def plot_score_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['Overall'], bins=10, kde=True)
    plt.title('Distribution of Essay Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()
