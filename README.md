# Machine Learning Labs

Welcome to the **Machine Learning Labs** repository! This collection includes hands-on lab exercises and implementations covering various fundamental and advanced topics in Machine Learning. These labs are designed to deepen your understanding of core ML concepts through practical coding experiences using Python and popular ML libraries.

## 📁 Repository Structure

The repository is organized by lab number. Each folder typically contains:
- Jupyter Notebooks or Python scripts
- Sample datasets (if needed)
- Supporting files (e.g., plots, reports)
*****************************************************************************************************************
**Lab 01: **Environment Setup & GitHub Initialization****
This lab focuses on setting up the necessary tools and environments for running machine learning projects efficiently.

🔧 Tasks Completed
Installed a Python IDE (such as VS Code, or Jupyter Notebook).
Created a GitHub account for version control and sharing code.
Set up Jupyter Notebook for running ML experiments interactively.
Created and configured a virtual environment to manage project dependencies.

**📂 Folder Contents:**
Lab01_Setup_and_Environment/
─ setup_instructions.md
─ jupyter_setup.png (optional screenshot of Jupyter working)
─ environment_setup.txt (list of installed packages or steps followed)

**📝 Notes**
Jupyter notebooks are highly recommended for lab work due to their support for inline visualization and markdown.
Using virtual environments (venv or conda) ensures dependency isolation and project reproducibility.
Version control with Git and GitHub will be used throughout this course/lab series.
*****************************************************************************************************************

**Lab 02**: **Introduction to Supervised Learning – Linear Regression****
This lab introduces the fundamental concepts of Supervised Learning using Linear Regression. You will implement a simple linear regression model from scratch and also using scikit-learn.

📌 Topics Covered
Understanding the difference between supervised and unsupervised learning.
Implementing Linear Regression:
Using NumPy (manual implementation)
Using scikit-learn (automated approach)
Evaluating model performance (e.g., MSE, R² score)
Visualizing data and regression lines with Matplotlib

**📁 Folder Structure**
Lab02_Linear_Regression/
─ Lab_2_detail.ipynb
─ dataset.csv (if any used)
─ output_plots/ (optional: graphs generated)
**🧪 How to Run**
1. Open the Jupyter Notebook:
          jupyter notebook Lab_2_detail.ipynb
2. Step through the cells to:
          . Load and explore the dataset
          . Train and test a Linear Regression model
          . Evaluate and visualize results
*********************************************************************************************************************

**Lab 03: Data Handling with Pandas (and NumPy)****
 This lab focuses on mastering two essential Python libraries for data manipulation: NumPy and Pandas. These tools are foundational for any machine learning workflow.

📘 Part 1: Working with Pandas
Importing and exploring datasets
DataFrame creation and manipulation
Handling missing values
Sorting, filtering, and grouping
Statistical analysis with .describe(), .mean(), .median(), etc.

📦 Folder Structure:
  Lab03_Data_Manipulation/
─ Lab 3 pandas.ipynb
─ Lab 3 numpy.ipynb        
─ sample_dataset.csv      
********************************************************************************************************************

**Lab 4.1 – Missing Data Handling****
     Focuses on identifying and managing missing values in time-series data.
**Lab 4.2 – Outlier Identification and Treatment****
     Uses the IQR method to detect and fill outliers.
**Lab 4.3 – Introducing Holidays into Features****
     Enhances the dataset by incorporating holiday-related features.

# Lab 4 – Data Preprocessing in Machine Learning

This section of the **Machine Learning Labs** repository focuses on essential data preprocessing techniques to prepare datasets for robust machine learning modeling. Each lab explores a key preprocessing task using Python and popular libraries like `pandas`, `numpy`, and `matplotlib`.

**Lab Contents**
📘 Lab 4.1 – Handling Missing Data
- **Objective:** Identify and handle missing values in a dataset, particularly time-series data.
- **Key Concepts:** 
  - Missing value detection
  - Visualization of gaps in data
  - Imputation techniques: forward fill, backward fill, interpolation
- **Tools Used:** `pandas`, `matplotlib`

📘 Lab 4.2 – Outlier Identification and Treatment
- **Objective:** Detect and treat outliers using the Interquartile Range (IQR) method.
- **Key Concepts:**
  - Statistical detection of outliers
  - Data cleaning through replacement strategies
  - Visual inspection with boxplots
- **Tools Used:** `pandas`, `numpy`, `matplotlib`

📘 Lab 4.3 – Introducing Holidays into the Dataset
- **Objective:** Incorporate holiday data to enhance time-based models.
- **Key Concepts:**
  - Feature engineering with public holidays
  - Date and time processing
  - Merging external data sources
- **Tools Used:** `pandas`, `datetime`, `holidays`

**How to Run**
1. Install the required packages:
      pip install pandas numpy matplotlib holidays
2. Open each Jupyter notebook using:
   jupyter notebook

*************************************************************************************************************************

**Lab 5.1 – Feature Extraction****: 
          Focuses on generating informative features from raw data.
**Lab 5.2 – Correlation Analysis****: 
           Explores relationships between features using correlation techniques.


# Lab 5 – Feature Engineering and Correlation Analysis

Welcome to **Lab 5** of the Machine Learning Labs series. In this section, we explore critical steps in preparing data for machine learning models: extracting meaningful features and understanding their interrelationships through correlation analysis.

**Lab Contents**
🧠 Lab 5.1 – Feature Extraction
- **Objective:** Extract informative features from raw time-series or structured data to improve model performance.
- **Key Concepts:**
  - Feature creation using date/time components (e.g., hour, day of week)
  - Rolling statistics and lag features
  - Domain-specific feature engineering
- **Tools Used:** `pandas`, `numpy`

🔍 Lab 5.2 – Correlation Analysis
- **Objective:** Analyze the strength and direction of relationships between features using correlation matrices.
- **Key Concepts:**
  - Pearson correlation coefficient
  - Heatmaps for visualizing correlations
  - Feature selection insights based on multicollinearity
- **Tools Used:** `pandas`, `seaborn`, `matplotlib`

**How to Run**
1. Install the required packages:
      pip install pandas numpy seaborn matplotlib
************************************************************************************************************************

**Lab 6**  focuses on:
   **Normalization**
   **One-Hot Encoding**
   **Cyclic Feature Encoding**

# Lab 6 – Feature Scaling and Encoding

**Lab 6** in the Machine Learning Labs series emphasizes the importance of transforming data through normalization and encoding techniques to make it suitable for machine learning algorithms.

**Lab Overview**

⚙️ Normalization
- **Objective:** Scale numeric features to a common range.
- **Key Techniques:**
  - Min-Max Scaling
  - Z-score Standardization
- **Why it matters:** Ensures that features contribute equally to model learning, especially important for distance-based algorithms like KNN or gradient descent optimization.

🧩 One-Hot Encoding
- **Objective:** Convert categorical variables into a format that can be provided to ML algorithms.
- **Key Techniques:**
  - `pandas.get_dummies()`
  - Handling high-cardinality categorical features

🔁 Cyclic Feature Encoding
- **Objective:** Properly represent cyclical features such as time of day or day of the week.
- **Key Techniques:**
  - Transforming with sine and cosine functions to preserve circular relationships

**Tools Used**
- `pandas`
- `numpy`
- `sklearn.preprocessing`
- `matplotlib`

**How to Run**
1. Install required packages:
      pip install pandas numpy scikit-learn matplotlib
2. Open the Jupyter notebook
********************************************************************************************************************

**Lab 7 – MLP (Multi-Layer Perceptron)**
This lab introduces the implementation and training of neural networks using a Multi-Layer Perceptron (MLP), likely involving a framework like TensorFlow or PyTorch.

# Lab 7 – Multi-Layer Perceptron (MLP)

Welcome to **Lab 7** of the Machine Learning Labs series. In this lab, you'll build and train a **Multi-Layer Perceptron (MLP)** — a foundational type of neural network used for supervised learning tasks such as regression and classification.

**Lab Objectives**

- Understand the structure and components of an MLP
- Implement and train a simple neural network model
- Evaluate model performance on training and testing data
- Visualize learning progress (e.g., loss over epochs)

 **Key Concepts**
- **Feedforward Neural Networks**
- **Backpropagation and Gradient Descent**
- **Activation Functions** (e.g., ReLU, sigmoid)
- **Loss Functions** (e.g., MSE, Cross-Entropy)
- **Model Evaluation Metrics**

## Tools & Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow` or `torch` (depending on the framework used in the notebook)

**How to Run**
1. Install the required packages (example for TensorFlow):
      pip install numpy pandas matplotlib tensorflow

**Prerequisites**
Solid understanding of basic ML concepts
Familiarity with Python and Jupyter Notebooks
Introductory knowledge of neural networks

**🤖 Note:** 
The MLP is a building block for deeper architectures. Mastering this helps in understanding more complex models like CNNs and RNNs

*****************************************************************************************************************************

**Lab 8 – 1D CNN**
This lab focuses on **1D Convolutional Neural Networks**, typically used for sequential data like time series, signal processing, or NLP tasks.

# Lab 8 – 1D Convolutional Neural Networks (1D CNN)
Welcome to **Lab 8** of the Machine Learning Labs series. This lab introduces **1D Convolutional Neural Networks (CNNs)**, which are well-suited for processing sequential data such as time-series signals or sensor data.

**Lab Objectives**

- Understand the architecture and operation of 1D CNNs
- Apply convolutional layers to extract features from sequential input
- Build and train a 1D CNN model using real-world or synthetic data
- Evaluate and visualize model performance

**Key Concepts**
- **1D Convolutions and Kernels**
- **Feature Maps and Filters**
- **Pooling Layers (Max Pooling)**
- **Activation Functions (e.g., ReLU)**
- **Model Evaluation and Visualization**

**Tools & Libraries Used**
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow.keras` or `torch.nn` (depending on the framework used in the notebook)

**How to Run**
1. Install required libraries (example for TensorFlow):
   pip install numpy pandas matplotlib tensorflow
2. Launch the notebook:
   jupyter notebook

**Prerequisites**
Understanding of neural networks and MLPs
Familiarity with convolution operations
Basic experience with a deep learning framework (TensorFlow or PyTorch)
****************************************************************************************************************************

**Lab 9 – RNN**
This lab covers **Recurrent Neural Networks**, a fundamental architecture for modeling sequential and time-series data.

# Lab 9 – Recurrent Neural Networks (RNN)

Welcome to **Lab 9** of the Machine Learning Labs series. In this lab, you will explore **Recurrent Neural Networks (RNNs)**, a class of neural networks ideal for handling sequential data where context and order are crucial.

**Lab Objectives**

- Understand the architecture and flow of RNNs
- Implement and train an RNN for time-series or sequence modeling tasks
- Explore vanishing gradients and limitations of vanilla RNNs
- Visualize the learning and performance of the network

**Key Concepts**

- **Sequential Data Modeling**
- **Recurrent Connections and Hidden States**
- **Backpropagation Through Time (BPTT)**
- **Challenges in RNNs** (e.g., vanishing gradients)
- (Optional) **Introduction to LSTM/GRU** if covered in the notebook

**Tools & Libraries Used**

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow.keras` or `torch.nn` for RNN layers

**How to Run**

1. Install the required libraries:
   pip install numpy pandas matplotlib tensorflow
2. Start Jupyter Notebook

**🔁 Insight:**
RNNs are essential for tasks like language modeling, sequence prediction, and time-series forecasting — this lab is a gateway to more advanced architectures like LSTM and GRU.

**************************************************************************************************************************

**Lab 10 – LSTM****
This lab introduces **Long Short-Term Memory (LSTM) networks**, an advanced form of RNNs designed to better capture long-range dependencies in sequential data.

# Lab 10 – Long Short-Term Memory Networks (LSTM)
Welcome to **Lab 10** of the Machine Learning Labs series. In this lab, you’ll explore **LSTM (Long Short-Term Memory)** networks — a powerful type of recurrent neural network designed to overcome the vanishing gradient problem and learn from long sequences.

**Lab Objectives**

- Understand the structure and functionality of LSTM cells
- Implement an LSTM model for time-series prediction or sequence classification
- Compare LSTM performance with vanilla RNNs
- Visualize model performance over training

**Key Concepts**

- **LSTM Cell Architecture**
  - Input, Forget, and Output Gates
  - Cell States and Hidden States
- **Sequential Data Handling**
- **Model Training and Evaluation**
- **(Optional)** Advanced tuning or stacked LSTMs

**Tools & Libraries Used**

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow.keras` or `torch.nn` for LSTM layers

**How to Run**

1. Install the required packages:
   pip install numpy pandas matplotlib tensorflow
2. for PyTorch:
   pip install numpy pandas matplotlib torch

******************************************************************************************************

**Lab 11 – Image Dataset Making**
This lab appears to focus on preparing and organizing image data for computer vision tasks—an essential step before training models like CNNs.

# Lab 11 – Image Dataset Preparation
Welcome to **Lab 11** of the Machine Learning Labs series. In this lab, you'll learn how to build and prepare an image dataset for machine learning and deep learning applications.

**Lab Objectives**

- Collect, organize, and preprocess image datasets
- Understand directory structures compatible with deep learning frameworks (e.g., Keras, PyTorch)
- Perform image resizing, normalization, and augmentation
- Save processed datasets for model training

**Key Concepts**

- **Image Acquisition** (from local files, web, or datasets)
- **Directory Structuring** (class-wise folders)
- **Image Preprocessing**
  - Resizing
  - Normalization
  - Format conversion
- **Data Augmentation** (optional, if included)

**Tools & Libraries Used**

- `os`, `shutil`
- `PIL` or `cv2` (OpenCV)
- `numpy`
- `matplotlib`
- `tensorflow.keras.preprocessing` or `torchvision.transforms` (if applicable)

**How to Ru**n

1. Install the required libraries:
   pip install numpy matplotlib pillow opencv-python
(or add tensorflow or torchvision if using advanced tools)

***************************************************************************************************************************

**Lab 12 – Design Your Own CNN!** 
This final lab likely focuses on creating and training a custom **Convolutional Neural Network (CNN)** for image classification or a similar task.

# Lab 12 – Design Your Own Convolutional Neural Network (CNN)
Welcome to **Lab 12** of the Machine Learning Labs series. In this final lab, you'll bring together everything you've learned to **design, implement, and train your own custom CNN** for an image classification task.

**Lab Objectives**

- Design a custom CNN architecture from scratch
- Train the CNN on a real-world or prepared image dataset
- Evaluate and interpret model performance using metrics and visualizations
- Compare your model’s performance to prebuilt architectures

**Key Concepts**

- **Convolutional Layers**
- **Pooling Layers (Max/Average Pooling)**
- **Flattening and Fully Connected Layers**
- **Activation Functions (ReLU, softmax)**
- **Training and Evaluation Metrics (Accuracy, Loss, Confusion Matrix)**

**Tools & Libraries Used**

- `tensorflow.keras` or `torch.nn` for building and training the model
- `numpy`
- `matplotlib`, `seaborn` (for visualizations)
- `sklearn` (optional, for evaluation)

**How to Run**

1. Install the required packages:
   pip install numpy matplotlib tensorflow
2. Open Lab 12 – Design Your Own CNN and follow the step-by-step process to build and train your network.

****************************************************************************************************************

**Lab 13 – Augmentation and Keras ImageDataGenerator**
This lab focuses on **image data augmentation**, a powerful technique to improve model generalization, using Keras'

# Lab 13 – Image Augmentation with Keras ImageDataGenerator
Welcome to **Lab 13** of the Machine Learning Labs series. This lab introduces **data augmentation techniques** using Keras' `ImageDataGenerator` to improve model robustness and reduce overfitting in image-based deep learning tasks.

**Lab Objectives**

- Understand the importance of data augmentation in deep learning
- Apply various augmentation techniques using `ImageDataGenerator`
- Visualize augmented images to understand transformations
- Integrate augmentation into the model training pipeline

**Key Concepts**

- **Overfitting Prevention**
- **On-the-fly Data Augmentation**
- **Transformations:**
  - Rotation
  - Zoom
  - Shift
  - Horizontal/Vertical Flip
  - Brightness/Contrast Adjustment
- **Flow from Directory and Flow from Data**

**Tools & Libraries Used**

- `tensorflow.keras.preprocessing.image.ImageDataGenerator`
- `matplotlib` (for visualization)
- `numpy`, `os`, `shutil`

**How to Run**

1. Install the required libraries:
   pip install tensorflow numpy matplotlib

2. Launch Jupyter Notebook:
   jupyter notebook

4. Open and run Lab 13 – Image Augmentation with ImageDataGenerator.

***********************************************************************************************************
📘**Conclusion**
This concludes the Machine Learning Labs series documented in this repository. Spanning from data preprocessing and feature engineering to advanced neural network architectures and image data augmentation, these 13 labs provide a structured and practical path through core machine learning and deep learning topics. Each lab builds on the previous one, offering hands-on experience with real-world tools and libraries like TensorFlow, Keras, and PyTorch. All labs are described in this single README.md file to provide a centralized and accessible guide for learners and practitioners. Whether you're a student, educator, or self-learner, this lab series aims to bridge theory and practice — equipping you with the skills needed to build intelligent systems with confidence.

