import matplotlib.pyplot as plt
import seaborn as sns

def correlation_(credit_data, numeric_exclude=None, figsize=(12, 10), annot=True, fmt=".2f", cmap='jet'):

    numeric_columns = credit_data.select_dtypes(include='number')
    if numeric_exclude:
        numeric_columns = numeric_columns.drop(numeric_exclude, axis=1)
    
    correlation_matrix = numeric_columns.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, fmt=fmt, cmap=cmap)
    plt.title('Correlation Heatmap')
    plt.show()


def scatter_plot_grouped(credit_data, group_column, x_column, y_column, palette='viridis', alpha=0.7, figsize=(12, 8)):
    """
    Veri ve sütun isimlerini parametre alır, scatter plot olarak geriye döner.

    Parameters:
    - data (pd.DataFrame): Scatter plot çizilecek DataFrame.
    - group_column (str): DataFrame'i gruplamak için kullanılacak sütun adı.
    - x_column (str): x ekseni için kullanılacak sütun adı.
    - y_column (str): y ekseni için kullanılacak sütun adı.
    - palette (str or list): seaborn paleti veya renk listesi.
    - alpha (float): nokta şeffaflığı (0.0 ile 1.0 arasında).
    - figsize (tuple): scatter plot boyutu.

    Returns:
    - None
    """
    # DataFrame'i belirli sütuna göre grupla ve ortalamalarını al
    grouped_data = credit_data.groupby(group_column)[[x_column, y_column]].mean().reset_index()

    # Scatter plot çizimi
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x_column, y=y_column, hue=group_column, data=grouped_data, palette=palette, alpha=alpha)
    plt.title(f'{x_column} vs. {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()


def plot_percentage_distribution(credit_data, column, figsize=(10, 6), palette='viridis', rotation=45, ha='right'):
    """
    Verilen DataFrame'deki belirli bir sütunun değerlerinin yüzde dağılımını gösteren bar plot çizer.

    Parameters:
    - data (pd.DataFrame): Bar plot çizilecek DataFrame.
    - column (str): Yüzde dağılımı gösterilecek sütun adı.
    - figsize (tuple): Bar plot boyutu.
    - palette (str or list): Seaborn paletti veya renk listesi.
    - rotation (float): X ekseni etiketlerinin dönme açısı.
    - ha (str): X ekseni etiketlerinin hizalaması.

    Returns:
    - None
    """
    # Sütunun değerlerinin yüzde dağılımını hesapla
    percentage_distribution = credit_data[column].value_counts(normalize=True) * 100

    # Bar plot çizimi
    plt.figure(figsize=figsize)
    sns.barplot(x=percentage_distribution.index, y=percentage_distribution.values, palette=palette)
    plt.title(f'Percentage Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=rotation, ha=ha)
    plt.show()

def pie_chart_distribution(credit_data, column, figsize=(8, 8), palette=None, startangle=140):
    """
    Verilen DataFrame'deki belirli bir sütundaki değerlerin dağılımını pasta grafiği olarak gösterir.

    Parameters:
    - data (pd.DataFrame): Pasta grafiği çizilecek DataFrame.
    - column (str): Dağılımı gösterilecek sütun adı.
    - figsize (tuple): Pasta grafiği boyutu.
    - palette (str or list): Seaborn paletti veya renk listesi.
    - startangle (float): Pasta grafiğinin başlangıç açısı.

    Returns:
    - None
    """
    # Sütundaki değerlerin sayısını hesapla
    value_counts = credit_data[column].value_counts()

    # Renk paletini belirle
    custom_palette = sns.color_palette(palette) if palette else None

    # Pasta grafiği çizimi
    plt.figure(figsize=figsize)
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=startangle, colors=custom_palette)
    plt.title(f'Distribution of {column}')
    plt.show()

def boxplot_comparison(data, x, y, figsize=(12, 8), palette=None, title=None):
    """
    Verilen DataFrame'deki iki sayısal sütun arasındaki ilişkiyi kutu grafiği ile gösterir.

    Parameters:
    - data (pd.DataFrame): Kutu grafiği çizilecek DataFrame.
    - x (str): Kutu grafiğinde x ekseni olarak kullanılacak sütun adı.
    - y (str): Kutu grafiğinde y ekseni olarak kullanılacak sütun adı.
    - figsize (tuple): Kutu grafiği boyutu.
    - palette (str or list): Seaborn paletti veya renk listesi.
    - title (str): Kutu grafiği başlığı.

    Returns:
    - None
    """
    # Renk paletini belirle
    custom_palette = sns.color_palette(palette) if palette else None

    # Kutu grafiği çizimi
    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=x, y=y, palette=custom_palette)
    plt.title(title if title else f'Boxplot of {y} vs {x}')
    plt.show()