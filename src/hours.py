from util import load_data
import matplotlib.pyplot as plt

X, y = load_data()
X_fraud = X[y == 1]
X_genuine = X[y == 0]

# Create new numpy array representing the "hour" of the data relative to the starting time
fraud_hour = (X_fraud.Time / 3600) % 24
genuine_hour = (X_genuine.Time / 3600) % 24

plt.hist([fraud_hour, genuine_hour], label=['Fraud', 'Genuine'], bins=24, density=True)
plt.legend(loc='upper right')
plt.xlabel('Hour bucket')
plt.ylabel('Density')
plt.xlim([0, 24])
plt.title('Histogram comparing the density of fraud\n and genuine transactions per hour bucket')
plt.show()