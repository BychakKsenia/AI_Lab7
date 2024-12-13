import datetime
import json
import numpy as np
import yfinance as yf
from sklearn import covariance, cluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

input_file = 'company_symbol_mapping.json'

# Зчитуємо мапу символів компаній
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())
symbols, names = np.array(list(company_symbols_map.items())).T

# Визначаємо діапазон дат для отримання котирувань
start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)

# Отримуємо історичні котирування через yfinance
quotes = []
for symbol in symbols:
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if not data.empty:  # Перевіряємо, чи є дані
            quotes.append(data)
        else:
            print(f"No data found for {symbol}")
            quotes.append(None)  # Додаємо None, якщо дані відсутні
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        quotes.append(None)  # Додаємо None, якщо сталася помилка

# Фільтруємо порожні котирування
quotes = [quote for quote in quotes if quote is not None]

# Якщо після фільтрації немає котирувань, вивести помилку
if not quotes:
    raise ValueError("No valid data was found for any of the symbols.")

# Перевірка на однаковість розміру всіх котирувань
min_length = min(len(quote) for quote in quotes)  # Знаходимо мінімальну кількість рядків серед котирувань
quotes = [quote.iloc[:min_length] for quote in quotes]  # Обрізаємо котирування до мінімальної довжини

# Збираємо ціни відкриття та закриття
opening_quotes = np.array([quote['Open'].values for quote in quotes]).astype(float)  # Замінили np.float на float
closing_quotes = np.array([quote['Close'].values for quote in quotes]).astype(float)  # Замінили np.float на float

# Обчислюємо різницю між цінами закриття та відкриття
quotes_diff = closing_quotes - opening_quotes

# Перевіряємо розмірність перед нормалізацією
print("Shape of quotes_diff before normalization:", quotes_diff.shape)

# Нормалізація даних
X = quotes_diff.reshape(-1, len(symbols))  # Reshape to (681, 4)
std_devs = X.std(axis=0)
X /= np.where(std_devs != 0, std_devs, 1)  # If std dev is zero, don't divide

# Перевіряємо розмірність даних перед подачею в модель
print("Shape of X before fitting:", X.shape)

# Створюємо модель для кластеризації
edge_model = covariance.GraphicalLassoCV()

# Навчаємо модель
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Виконуємо кластеризацію
_, labels = cluster.affinity_propagation(edge_model.covariance_)

# Виводимо результати кластеризації
num_labels = labels.max()
for i in range(num_labels + 1):
    print("Cluster", i + 1, "==>", ', '.join(names[labels == i]))

# Візуалізація кластерів

# Знижуємо розмірність до 2D за допомогою PCA для візуалізації
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.T)  # Транспонуємо X для роботи з ознаками

# Створюємо графік
plt.figure(figsize=(10, 8))

# Для кожного кластеру використовуємо окремий колір
for i in range(num_labels + 1):
    cluster_points = X_2d[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

    # Додаємо назви компаній в кластері
    for j in range(len(cluster_points)):
        plt.text(cluster_points[j, 0], cluster_points[j, 1], names[labels == i][j], fontsize=9)

plt.title('Clusters of Companies based on Stock Price Variation')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()
