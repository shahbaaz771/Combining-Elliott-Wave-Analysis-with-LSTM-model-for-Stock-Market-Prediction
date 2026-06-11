# Combining Elliott Wave Analysis with LSTM for Stock Market Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Finance](https://img.shields.io/badge/Domain-Stock%20Market-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Deep%20Learning-red)

## Overview

Stock market prediction remains one of the most challenging problems in financial analytics due to the highly volatile and non-stationary nature of market data.

This project presents a hybrid approach that combines:

* **Elliott Wave Theory (EWT)** for technical analysis
* **Long Short-Term Memory (LSTM)** neural networks for time-series forecasting

The objective is to identify Elliott Wave structures in historical stock data and use these wave patterns as inputs to an LSTM model for predicting future market movements and potential trading opportunities.

The model was evaluated using historical **Apple Inc. (AAPL)** stock data and demonstrated improved predictive performance by integrating technical market structure information with deep learning.

---

## Motivation

Traditional machine learning models often struggle with:

* Market volatility
* Non-linear price movements
* Investor sentiment cycles
* Long-term temporal dependencies

Elliott Wave Theory provides a framework for identifying recurring market patterns driven by investor psychology, while LSTM networks excel at learning sequential dependencies in financial time-series data.

By combining both approaches, the model attempts to improve forecasting accuracy and identify potential entry and exit points for trading decisions.

---

## Methodology

### Step 1: Data Collection

Historical stock data is collected using:

* Yahoo Finance API
* yFinance
* pandas-datareader

Example stock:

```text
AAPL (Apple Inc.)
```

---

### Step 2: Elliott Wave Analysis

The project uses the `taew` library to identify Elliott Wave structures.

The algorithm:

1. Detects upward Elliott Waves
2. Extracts wave turning points
3. Identifies retracement zones
4. Determines potential:

* Buying points
* Selling points

based on Fibonacci retracement relationships.

---

### Step 3: Data Preprocessing

The extracted Elliott Wave points are:

* Cleaned
* Normalized using MinMaxScaler
* Converted into supervised learning sequences

This transformation prepares the data for deep learning training.

---

### Step 4: LSTM Model

The forecasting model consists of:

* Three stacked LSTM layers
* Dropout regularization
* Dense output layer

Architecture:

```text
Input Sequence
      ↓
LSTM Layer (50 Units)
      ↓
Dropout (0.2)
      ↓
LSTM Layer (50 Units)
      ↓
Dropout (0.2)
      ↓
LSTM Layer (50 Units)
      ↓
Dropout (0.2)
      ↓
Dense Layer
      ↓
Predicted Elliott Wave Point
```

---

## Features

* Historical stock data acquisition
* Elliott Wave detection
* Buy and sell signal identification
* Data normalization and preprocessing
* LSTM-based forecasting
* Visualization of wave structures
* Future price prediction
* Model evaluation using error metrics

---

## Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* TensorFlow / Keras
* Scikit-Learn
* Yahoo Finance (yFinance)
* pandas-datareader
* taew

---

## Project Structure

```text
Combining-Elliott-Wave-Analysis-with-LSTM-model-for-Stock-Market-Prediction/
│
├── Elliot_wave_and_LSTM.ipynb
├── Elliot_wave_and_LSTM.py
├── README.md
│
└── results/
    ├── elliott_wave_detection.png
    ├── stock_prediction.png
    └── evaluation_metrics.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shahbaaz771/Combining-Elliott-Wave-Analysis-with-LSTM-model-for-Stock-Market-Prediction.git
```

Navigate to the project directory:

```bash
cd Combining-Elliott-Wave-Analysis-with-LSTM-model-for-Stock-Market-Prediction
```

Install dependencies:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn yfinance pandas-datareader taew
```

---

## Usage

Run the Python script:

```bash
python Elliot_wave_and_LSTM.py
```

Or open the notebook:

```bash
jupyter notebook Elliot_wave_and_LSTM.ipynb
```

---

## Workflow

```text
Historical Stock Data
            ↓
    Elliott Wave Analysis
            ↓
  Wave Point Extraction
            ↓
      Data Scaling
            ↓
      LSTM Training
            ↓
 Future Wave Prediction
            ↓
 Trading Signal Generation
```

---

## Evaluation Metrics

The model is evaluated using:

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)

These metrics help measure forecasting accuracy and prediction reliability.

---

## Results

The hybrid EWT-LSTM approach demonstrated improved forecasting performance compared to using standalone price prediction methods.

Key observations:

* Better identification of retracement points
* Improved trend recognition
* Enhanced market phase understanding
* More informative buy/sell signals

---

## Limitations

* Stock markets are inherently unpredictable
* Elliott Wave labeling can be subjective
* Performance may vary across different stocks and market conditions
* Historical performance does not guarantee future results

---

## Future Enhancements

* Multi-stock forecasting
* Real-time prediction dashboard
* Transformer-based forecasting models
* Sentiment analysis integration
* News and social media signal incorporation
* Portfolio optimization module

---

## Learning Outcomes

This project demonstrates:

* Financial time-series forecasting
* Technical analysis automation
* Deep learning with LSTM networks
* Data preprocessing and feature engineering
* Quantitative finance applications
* Algorithmic trading concepts

---

## Author

**Shahbaaz Ahmed Sadiq**

Computer Science & Engineering Graduate

GitHub: https://github.com/shahbaaz771

---

## Disclaimer

This project is intended for educational and research purposes only.

The predictions generated by the model should not be considered financial advice or investment recommendations.

---

## License

MIT License
