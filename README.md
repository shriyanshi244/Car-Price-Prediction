# 🚗 Car Price Prediction using Machine Learning

A machine learning project that predicts the **selling price of used cars** using vehicle-related features such as brand, mileage, engine capacity, maximum power, kilometers driven, fuel type, transmission, seller type, and ownership history.

The project uses **Linear Regression** for price prediction and includes data preprocessing, exploratory data analysis, feature encoding, model training, and model serialization.

## 📌 Project Overview

The objective of this project is to build a machine learning model that can estimate the selling price of a used car based on its available specifications.

### Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Duplicate & Missing Value Removal
   ↓
Feature Cleaning
   ↓
Categorical Encoding
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Linear Regression
   ↓
Model Training
   ↓
Save Model as model.pkl
```

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation and preprocessing
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Exploratory data analysis
* **Scikit-learn** – Machine learning and evaluation
* **Pickle** – Saving the trained model
* **Jupyter Notebook** – Development environment

## 📂 Project Structure

```text
Car-Price-Prediction/
│
├── Car_Price_Model.ipynb
├── Cardetails.csv
├── model.pkl
└── README.md
```

## 📊 Dataset

The project uses a CSV dataset named:

```text
Cardetails.csv
```

The dataset contains information about used cars, including features such as:

* Car name/brand
* Year
* Selling price
* Kilometers driven
* Fuel type
* Seller type
* Transmission
* Ownership
* Mileage
* Engine
* Maximum power
* Torque

The `torque` column is removed during preprocessing.

## 🧹 Data Preprocessing

The following preprocessing operations are performed:

1. Removal of the `torque` column.
2. Removal of duplicate records.
3. Removal of rows containing missing values.
4. Extraction of the car brand from the car name.
5. Cleaning of `mileage`, `max_power`, and `engine` values.
6. Conversion of categorical variables into numerical values.

### Categorical Encoding

The notebook converts categorical features into numerical representations.

For example:

```text
Transmission:
Manual     → 1
Automatic  → 2
```

```text
Fuel:
Diesel  → 1
Petrol  → 2
LPG     → 3
CNG     → 4
```

Ownership categories are also converted into numerical values.

## 📈 Exploratory Data Analysis

Several visualizations are generated to understand the dataset and relationships between features.

### Visualizations Included

* Correlation Heatmap
* Fuel Type Distribution
* Selling Price vs. Kilometers Driven
* Selling Price by Transmission Type
* Average Selling Price by Ownership
* Selling Price Distribution

These visualizations help identify relationships and patterns within the car-price data.

## 🤖 Machine Learning Model

The project uses:

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
```

The target variable is:

```text
selling_price
```

All other available processed features are used as input features.

## 🔀 Train-Test Split

The dataset is divided into training and testing sets using an **80:20 split**.

```python
train_test_split(
    input_data,
    output_data,
    test_size=0.2,
    random_state=42
)
```

* **80%** → Training data
* **20%** → Testing data

## 💾 Model Saving

After training, the trained Linear Regression model is saved using Pickle:

```python
pk.dump(model, open('model.pkl', 'wb'))
```

This creates:

```text
model.pkl
```

The saved model can later be loaded into another Python application for prediction.

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Car-Price-Prediction.git
```

### 2. Navigate to the Project

```bash
cd Car-Price-Prediction
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Add the Dataset

Make sure `Cardetails.csv` is located in the same directory as the notebook.

### 5. Run the Notebook

```bash
jupyter notebook
```

Open:

```text
Car_Price_Model.ipynb
```

and run the cells.

## 📌 Model Output

After successful execution, the trained model is stored as:

```text
model.pkl
```

This file can be reused for making predictions without retraining the model.

## 🔮 Future Improvements

Possible improvements for this project include:

* Compare Linear Regression with Random Forest, XGBoost, and other regression algorithms.
* Add proper model evaluation using **RMSE** and **R² score**.
* Apply feature scaling where appropriate.
* Use one-hot encoding for categorical variables.
* Perform hyperparameter tuning for advanced models.
* Build a web interface using **Flask** or **Streamlit**.
* Deploy the trained model as a web application.
* Add an interactive car-price prediction form.

## 👩‍💻 Author

**Shriyanshi**

B.Tech – Computer Science & Engineering

---

⭐ If you found this project useful, consider giving the repository a star!
