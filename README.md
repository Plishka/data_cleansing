# User Rating Calculation Project

## Overview

This project demonstrates a comprehensive process of data cleaning, feature engineering, rating calculation based on specific criteria using the User Application dataset, and analysis using Python and popular data science libraries such as Pandas, Seaborn, and Matplotlib. The final visualization provides insights into the average rating of applications over time, aiding in better decision-making and analysis.

## Data Cleaning and Preparation

### Initial Data

The initial dataset, `applications.csv`, contains the following columns:

- `Applied at`: The date of application
- `Amount`: The amount applied for
- `Age`: The age of the applicant
- `Gender`: The gender of the applicant
- `Industry`: The industry of the applicant
- `Marital status`: The marital status of the applicant
- `External Rating`: The external rating of the applicant
- `Education level`: The education level of the applicant
- `Location`: The location of the applicant
- `applicant_id`: Unique identifier for each applicant

### Data Cleaning Steps

1. **Remove Duplicates**: Removed duplicate rows based on the `applicant_id`.
2. **Fill Missing Values**:
    - `External Rating` filled with zeros.
    - `Education level` filled with "Середня".
3. **Merge Industry Data**: Added industry ratings from the `industries.csv` file.

## Rating Calculation

The application rating is calculated based on several criteria:

1. **Base Criteria**:
    - The rating is zero if the `Amount` value is missing or `External Rating` equals zero.
2. **Scoring Components**:
    - 20 points if the applicant's age is between 35 and 55.
    - 20 points if the application was submitted on a weekday.
    - 20 points if the applicant is married.
    - 10 points if the applicant is located in Kyiv or the Kyiv region.
    - `Score` value from the `industries.csv` table (0 to 20 points).
    - 20 points if `External Rating` is greater than or equal to 7.
    - 20 points subtracted if `External Rating` is less than or equal to 2.

The final rating is adjusted to be between 0 and 100. Applications are considered accepted if the rating is greater than zero.

## Analysis and Visualization

### Grouping and Plotting

The data is grouped by the submission week, and the average rating of accepted applications for each week is plotted on a graph.

### Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
applications = pd.read_csv('applications.csv')
industries = pd.read_csv('industries.csv')

# Remove duplicates
applications.drop_duplicates('applicant_id', inplace=True)

# Fill missing values
applications['External Rating'].fillna(0, inplace=True)
applications['Education level'].fillna('Середня', inplace=True)

# Merge datasets
full_df = pd.merge(applications, industries, on='Industry', how='left')

# Convert dates
full_df['Applied at'] = pd.to_datetime(full_df['Applied at'], format='mixed')

# Calculate scores
full_df['Age Score'] = full_df['Age'].apply(lambda x: 20 if 35 <= x <= 55 else 0)
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
full_df['Applied Day'] = full_df['Applied at'].apply(lambda x: x.day_name())
full_df['Apply Score'] = full_df['Applied Day'].isin(weekdays) * 20
full_df['Marital Score'] = full_df['Marital status'].apply(lambda x: 20 if x == 'Married' else 0)
full_df['Location Score'] = full_df['Location'].apply(lambda x: 10 if x == 'Київ чи область' else 0)
full_df['External Score'] = full_df['External Rating'].apply(lambda x: 20 if x >= 7 else (-20 if x <= 2 else 0))

# Calculate total score
full_df['Total Score'] = full_df[['Score', 'Apply Score', 'Age Score', 'Marital Score', 'Location Score', 'External Score']].sum(axis=1)
full_df['Total Score'] = full_df['Total Score'].apply(lambda x: max(0, min(100, x)))
full_df.loc[full_df['External Rating'] == 0, 'Total Score'] = 0
full_df.loc[full_df['Amount'].isnull(), 'Total Score'] = 0

# Filter accepted applications
accepted = full_df[full_df['Total Score'] > 0]
accepted['Application Week'] = accepted['Applied at'].dt.isocalendar().week
accepted['Week Start'] = accepted['Applied at'].dt.to_period('W').apply(lambda y: y.start_time)

# Average rating by week
data = accepted.groupby('Week Start')['Total Score'].mean().reset_index()
data['Week Start'] = data['Week Start'].astype(str)

# Plot
plt.figure(figsize=(30, 10))
sns.barplot(x='Week Start', y='Total Score', data=data)
plt.title('Average Rating by Application Week', fontsize=30, fontweight='bold')
plt.xlabel(None, fontsize='xx-large')
plt.ylabel('AVG Score', fontsize='xx-large')
plt.xticks(rotation=45)
plt.show()
