# 🚀 Stock Price Prediction using LSTM

## 📚 Overview
This project demonstrates a stock price prediction model using Long Short-Term Memory (LSTM) neural networks. The model is trained on historical stock data from the 'all_stocks_5yr.csv' dataset and aims to forecast future stock prices using deep learning techniques.

## 🗂️ Project Structure
```
├── 📦 all_stocks_5yr.csv.zip  # Zipped dataset file
├── 📂 all_stocks_5yr.csv      # Unzipped dataset file
├── 📝 main.py                 # Script for data processing, training, and prediction
└── 📖 README.md               # Project documentation
```

## ✅ Prerequisites
Ensure the following libraries are installed:
- 🐍 Python 3.x
- 🟢 Pandas
- 🟡 NumPy
- 📊 Matplotlib
- 🎨 Seaborn
- 🔥 TensorFlow
- 🤖 Keras
- 🟦 Scikit-Learn

## 📊 Dataset
The dataset used in this project is 'all_stocks_5yr.csv' which contains stock data for multiple companies over a five-year period. It includes columns like:
- 📅 Date
- 💵 Open
- 💹 Close
- 📈 Volume
- 🏷️ Name (Company ticker)

## 🟢 How to Run
1. Extract the dataset from the ZIP file if not already done.
2. Execute the script using:
```bash
python main.py
```
3. The script performs the following steps:
- 🟢 Loads and preprocesses the data
- 📊 Visualizes stock prices and volume
- 🧠 Trains an LSTM model on Apple stock data
- 🔍 Makes predictions and visualizes results

## 🏆 Results
The model outputs metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE), and plots the predicted vs actual stock prices for Apple.

## 🌟 Future Enhancements
- 🛠️ Implement hyperparameter tuning for better accuracy.
- 🌀 Explore other deep learning architectures.
- 💻 Integrate a user interface for real-time predictions.

