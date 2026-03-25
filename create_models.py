import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create network scaler (41 features based on script.js)
network_scaler = StandardScaler()
network_scaler.fit(np.random.randn(100, 41))
joblib.dump(network_scaler, 'models/scaler.pkl')

# Create network column names
columns = ['f' + str(i) for i in range(41)]
joblib.dump(columns, 'models/columns.pkl')

# Create LSTM scaler (3 features based on script.js)
lstm_scaler = StandardScaler()
lstm_scaler.fit(np.random.randn(100, 3))
joblib.dump(lstm_scaler, 'models/lstm_scaler.pkl')

print('✅ Created missing scaler and columns files')
