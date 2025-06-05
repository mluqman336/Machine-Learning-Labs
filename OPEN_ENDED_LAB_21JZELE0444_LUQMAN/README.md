# LSTM-Based Energy Forecasting Using CityLearn Data

This project involves a multi-step data preprocessing and modeling pipeline for energy prediction using an LSTM (Long Short-Term Memory) neural network. The data is sourced from the [CityLearn](https://www.citylearn.net/) environment.

## 📁 Project Structure

```
├── data/
│   ├── pricing.csv
│   ├── weather.csv
│   ├── carbon_intensity.csv
│   ├── Building_2.csv
│   └── merged_data.csv        # Combined file with two columns
├── notebooks/
│   ├── Lab 4.1 - Missing Data
│   ├── Lab 4.2 - Outlier Identification (IQR)
│   ├── Lab 4.3 - Holiday Features
│   ├── Lab 5.1 - Feature Extraction
│   ├── Lab 5.2 - Correlation Analysis
│   ├── Lab 6 - Normalization, One-hot, Cyclic Encoding
│   └── Lab 10 - LSTM Modeling
└── README.md
```

## 📌 Step-by-Step Workflow

### 1. **Data Collection**
Four CSV files were downloaded from CityLearn:
- `pricing.csv`
- `weather.csv`
- `carbon_intensity.csv`
- `Building_2.csv`

### 2. **Data Merging**
These four files were merged into a single file, reducing the structure to only two columns for simplified modeling.

### 3. **Preprocessing Pipeline**

Each preprocessing step was handled in a modular Jupyter notebook:

- **Lab 4.1: Handling Missing Data**
  - Filled missing values using appropriate methods (e.g., interpolation or forward/backward fill).

- **Lab 4.2: Outlier Detection & Treatment**
  - Used the IQR method to identify and treat outliers.

- **Lab 4.3: Holiday Feature Engineering**
  - Incorporated holiday-based features to enhance temporal understanding.

- **Lab 5.1: Feature Extraction**
  - Extracted temporal, weather, and price-related features for modeling.

- **Lab 5.2: Correlation Analysis**
  - Analyzed feature correlation to identify impactful variables.

### 4. **Data Transformation**

- **Lab 6: Normalization & Encoding**
  - Applied MinMax normalization.
  - Performed one-hot encoding on categorical features.
  - Added cyclical encoding for time-based features (e.g., hour of day, day of week).

### 5. **Modeling: LSTM**

- **Lab 10: LSTM Neural Network**
  - Used a sequence-based LSTM model to predict future energy values.
  - Trained the model on the transformed dataset.
  - Evaluated using standard metrics like RMSE, MAE, and possibly MAPE.

## 🧪 Requirements

Install required packages using:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## 🧠 Key Learnings

- Sequential deep learning models like LSTM are effective for time-series energy forecasting.
- Preprocessing steps such as outlier handling, feature engineering, and normalization are critical for model performance.
- Feature selection based on correlation improves training efficiency.

## 👨‍💻 Author

**M. Luqman 0444**  
All labs and modeling pipelines were developed and executed by M. Luqman as part of OPEN_ENDED_Lab Series 4–10.
