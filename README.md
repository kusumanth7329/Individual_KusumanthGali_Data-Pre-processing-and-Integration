# Turbocharge Your Music Genre Detection Through Data Pre-processing and Integration

## 1. Introduction

**Title:**  
Turbocharge Your Music Genre Detection Through Data Pre-processing and Integration

**Research Question:**  
How do advanced data pre-processing techniques—such as imputation, normalization, encoding, and schema matching—enhance the accuracy of music genre classification models?

**Why This Question?**
- Music genre classification is a key problem in audio analysis and recommendation systems.
- Real-world datasets contain missing values, and inconsistencies, and require structured pre-processing.
- Effective pre-processing ensures improved model accuracy, minimizes bias, and enhances interpretability.
- Data integration is crucial in combining multiple sources to enrich classification models.

## 2. Theory and Background

### 2.1 Theoretical Foundation
Robust statistical methods form the backbone of our data pre-processing and integration strategy. We address missing values in skewed features—such as loudness (skewness = 2.1)—using median imputation (expressed as `df['loudness'].fillna(median)`).
<img width="174" alt="Screenshot 2025-02-16 at 6 48 11 PM" src="https://github.com/user-attachments/assets/3976d633-6afc-4f9b-87f3-4c64e2f510bb" />


This method preserves central tendencies while limiting outlier influence. For right-skewed variables like duration, we apply a log transformation:
<img width="216" alt="Screenshot 2025-02-16 at 6 50 09 PM" src="https://github.com/user-attachments/assets/7534f813-a048-45c2-b87f-a3656c01d8c8" />

which compresses extreme values (reducing skewness from 1.8 to 0.4) and brings the data closer to normality (Tukey, 1977). Standardization is then used to balance feature scales:
<img width="183" alt="Screenshot 2025-02-16 at 6 51 25 PM" src="https://github.com/user-attachments/assets/bf9892d6-e767-422e-9e9d-124c9b7cb942" />

Scaling features such as danceability and energy to a mean of 0 and a standard deviation of 1 with tools like StandardScaler() ensures equal contribution during model training (Géron, 2019). Finally, we integrate datasets by aligning them through common identifiers (song_id ↔ id), following schema matching principles (Guo et al., 2023); more complex cases may employ advanced methods like TF-IDF.
These approaches collectively enhance the robustness, scalability, and reproducibility of our Music Information Retrieval systems, effectively mitigating bias from noisy or incomplete data.


### 2.2 Literature Review
- **Music Information Retrieval (MIR):** Research in MIR highlights the importance of structured pre-processing for improving classification accuracy.
- **Imputation:** Rubin’s Missing Data (1987) established frameworks for handling gaps, while Liu et al. (2021) validated median imputation for skewed audio features.
- **Transformations:** Tukey’s Exploratory Data Analysis (1977) popularized log transforms; Zhang et al. (2022) demonstrated their efficacy in music feature engineering.
- **Standardization:** Géron’s Hands-On Machine Learning (2019) highlights its role in stabilizing gradient descent.
- **Schema Integration:** Bernstein et al. (2011) proposed automated schema matching, refined by Guo et al. (2023) using deep learning.

### 2.3 Key Data Science Concepts
1. **Feature Engineering:** Transforming raw audio metrics (e.g., tempo, loudness) into model-ready inputs via scaling, encoding, and interaction terms (e.g., energy * danceability).
2. **Handling Missing Data:** Imputation (Median/mean filling vs. deletion) and trade-offs (Simplicity vs. accuracy).
3. **Normalization vs. Standardization:** Normalization (min-max) bounds features to [0,1]; Standardization (z-score) centers them around zero.
4. **Categorical Encoding:** One-Hot: Used for mode and time_signature (low cardinality). Drop High-Cardinality: artist (150+ categories) removed to avoid sparse matrices.
5. **Data Integration:** Schema Matching: Aligning column names/data types across sources. Redundancy Handling: Removing duplicate entries.

## 3. Problem Statement

### 3.1 Detailed Problem Statement
Real-world music datasets used for genre classification often face several challenges, including incomplete data, skewed distributions (like right-skewed duration), schema mismatches (such as discrepancies between song_id and id columns), and high-cardinality categorical features (with over 150 unique artists). These issues can result in biased models and unreliable predictions.

To address these challenges, this work introduces a comprehensive pre-processing pipeline that utilizes statistically sound imputation methods—applying the median for skewed features and the mean for those that are normally distributed. It also employs log transformations to normalize skewed data, uses suitable encoding techniques for categorical variables (like one-hot encoding for low-cardinality features while excluding high-cardinality ones), and effectively aligns heterogeneous datasets through schema alignment. This approach results in a clean, structured dataset that is optimized for accurate, interpretable, and high-performing machine learning models.

### 3.2 Input-Output Format

**Input:**
- Raw Music Dataset: song_id, artist, album, release_year, and popularity.
- Audio Feature Data: Continuous variables such as danceability, energy, loudness, speechiness, and tempo.
- Genre Labels: Multi-class categorical variable representing music genres (e.g., Rock, Pop, Jazz, Classical).
- Challenges: Missing values, skewed distributions, inconsistent formatting, and non-numeric categorical data requiring transformation.

**Output:**
- Cleaned and Pre-Processed Dataset: Fully structured data with no missing values, scaled numerical attributes, and encoded categorical features.
- Optimized Feature Set for Classification: Features transformed for better interpretability and performance in machine learning models.
- Music Genre Classification Model: Predicts genres based on processed audio features, improving accuracy and efficiency.
- Interactive Insights and Visualizations: Feature importance analysis, genre distributions, and correlation heatmaps supporting decision-making.

### 3.3 Sample Inputs and Outputs

**Sample Input:**  
A raw dataset containing unprocessed song metadata and numerical features.
<img width="617" alt="Screenshot 2025-02-16 at 6 52 15 PM" src="https://github.com/user-attachments/assets/f14e0a5d-2ba9-474d-8968-ec58d9e9da6b" />

**Sample Output:**  
A fully processed dataset ready for machine learning models.
<img width="605" alt="Screenshot 2025-02-16 at 6 52 35 PM" src="https://github.com/user-attachments/assets/da3c7e8f-5a71-4d2b-8c99-288bc5a788ae" />
- Genres are encoded using One-Hot Encoding (genre_Blues, genre_R&B, etc.).
- All numerical attributes (danceability, energy, loudness) are scaled and normalized.
- Missing values are handled, ensuring model readiness.


## 4. Problem Analysis

### 4.1 Constraints and Challenges
- Handling Missing Data: Imputation techniques ensure data completeness.
- Normalization & Scaling: Log transformations applied to correct skewed distributions.
- Encoding Categorical Features: Label encoding and one-hot encoding applied for categorical variables.
- Data Integration: Merging datasets with different schemas requires careful alignment.

### 4.2 Approach
1. Data Exploration: Identify inconsistencies, outliers, and gaps.
2. Pre-processing Strategy: Apply structured cleaning and feature engineering.
3. Integration Strategy: Combine multiple datasets while preserving schema integrity.
4. Validation: Assess transformations through visualization and statistical checks.

### 4.3 Key Principles
1. Data Quality Assurance: Ensuring completeness, consistency, and accuracy.
2. Reproducibility: Implementing standardized pre-processing workflows.
3. Scalability: Designing solutions that work for large datasets.
4. Interpretability: Ensuring transformations retain meaningful insights.
5. Efficiency: Optimizing pre-processing steps for performance.

## 5. Solution Explanation

### 5.1 Step-by-Step Process
1. Load the Data: Import CSV files.
2. Understand the Data: View the summary statistics of the data.
3. Analyze the Data: Perform exploratory data analysis.
4. Data Integration: Merge datasets from multiple sources.
5. Data Transformation:
   - Apply log, square root, or Box-Cox transformation to normalize distributions.
6. Handle Missing Values:
   - Apply imputation or deletion as needed.
7. Normalize and Standardize:
   - Scale numerical features using min-max normalization or z-score standardization.
     <img width="450" alt="Screenshot 2025-02-16 at 6 53 35 PM" src="https://github.com/user-attachments/assets/664ef7c0-6d91-49be-b784-9caf26a1c945" />

8. Encode Categorical Variables:
   - Convert non-numeric fields using label encoding or one-hot encoding.
9. Validation and Visualization:
   - Validate data quality using histograms and correlation heatmaps.

### 5.2 Pseudocode

**Input:** Raw datasets genre_data.csv, meta_data.csv  
**Output:** Processed dataset processed_music_data.csv

<img width="561" alt="Screenshot 2025-02-16 at 6 54 05 PM" src="https://github.com/user-attachments/assets/0003d4b9-5f91-4b69-b227-37ab5cee9d0c" />

### 5.3 Logical Explanation
Our pre-processing pipeline ensures data correctness by aligning schemas for accurate merging, removing duplicates, and tailoring imputation strategies based on feature distribution. Skewed numerical features are imputed using the median and normalized with log transformations, while normally distributed features are imputed with the mean. Categorical variables are one-hot encoded or dropped based on their cardinality, and all numerical features are standardized using Z-scores. This systematic approach produces a clean, structured dataset optimized for reliable machine learning.

## 6. Results and Data Analysis

### 6.1 Well-Presented Results

**Handling Missing Values:**  
<img width="602" alt="Screenshot 2025-02-16 at 6 54 54 PM" src="https://github.com/user-attachments/assets/fdbdca12-2466-4c12-8fe2-95112390bcdb" />

<img width="663" alt="Screenshot 2025-02-16 at 6 55 31 PM" src="https://github.com/user-attachments/assets/b41674b3-48c9-4f29-a0c4-6bb56b6b0893" />

Fig. 1: Heatmap of Missing Values Before and After Imputation

**Feature Engineering and Standardization:**  
<img width="607" alt="Screenshot 2025-02-16 at 6 56 57 PM" src="https://github.com/user-attachments/assets/084899d4-3dbd-43cc-ae8c-9b31b46b5d13" />
<img width="648" alt="Screenshot 2025-02-16 at 6 57 14 PM" src="https://github.com/user-attachments/assets/8ecdcf82-9202-4537-abf0-8d212660e0e9" />

Fig. 2: Boxplot Comparison of Features Before and After Standardization

**Correlation Analysis:**  
<img width="561" alt="Screenshot 2025-02-16 at 6 57 52 PM" src="https://github.com/user-attachments/assets/fb0daa68-41e0-42b2-ba45-73e25051a1d3" />

Fig. 3: Correlation Heatmap of Features

### 6.2 Data Tables

**Table 1: Impact of Preprocessing**

<img width="568" alt="Screenshot 2025-02-16 at 6 58 17 PM" src="https://github.com/user-attachments/assets/b38dcff8-e35e-43f6-bd02-199f70d3c7e1" />

### 6.3 Insightful Discussion
- **Missing Value Imputation:**
  - **Choice of Imputation Method:**
    - Mean imputation was used for danceability and tempo (normally distributed).
    - Loudness, speechiness, duration, energy, and positiveness used median imputation due to skewness.
    - **Why It Matters:** Skewed features needed robust imputation to prevent bias in classification models.

- **Feature Engineering & Transformation:**
  - **Created interaction features:**
    - Energy × Danceability – Captures energetic but danceable songs.
    - Loudness × Speechiness – Differentiates vocal-heavy songs from instrumental tracks.
    - Tempo × Happening – Measures upbeat nature of songs.
  - **These new features improved feature correlations, enhancing model accuracy.**

### 6.4 Connection to Theoretical Background
Huber (1981) supports median imputation for handling skewed data, ensuring robustness against outliers. Music Information Retrieval (MIR) research highlights that feature engineering with interaction terms improves music classification. Géron (2019) emphasizes standardization as crucial for gradient-based optimization, enhancing model performance. Lastly, Guo et al. (2023) discuss schema matching & integration, ensuring data consistency for better machine learning outcomes.

### 6.5 Key Insights
- High correlation between energy and loudness.
- Certain genres have distinct tempo and danceability distributions.
- Normalization improves comparability across features.


## 7. References
- Pyle, D. (1999). Data Preparation for Data Mining. Morgan Kaufmann.
- Kazil, J., & Jarmul, K. (2016). Data Wrangling with Python. O'Reilly Media.
- Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques (3rd ed.). Morgan Kaufmann.
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
- IEEE Transactions on Knowledge and Data Engineering (Various Articles on Data Pre-processing and Integration).
- Musical Genre Classification of Audio Signals. IEEE Transactions on Speech and Audio Processing, 10(5), 293-302.
- Humphrey, E. J., Bello, J. P., & LeCun, Y. (2013). Feature Learning and Deep Architectures: New Directions for Music Informatics. Journal of Intelligent Information Systems, 41(3), 461-481.
