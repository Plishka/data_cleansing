# User Rating Calculation Project

## Overview

This project demonstrates a comprehensive process of data cleaning, feature engineering, rating calculation based on specific criteria using the User Application dataset, and analysis using Python and popular data science libraries such as Pandas, Seaborn, and Matplotlib. The final visualization provides insights into the average rating of applications over time, aiding in better decision-making and analysis.

## User Rating Calculation

### Overview and Conclusion

This project demonstrates a comprehensive process of data cleaning, feature engineering, rating calculation based on specific criteria using the User Application dataset, and analysis using Python and popular data science libraries such as Pandas, Seaborn, and Matplotlib. The final visualization provides insights into the average rating of applications over time, aiding in better decision-making and analysis.

### Process and Analysis

1. **Data Cleaning**:
    - Removed duplicates based on `applicant_id`.
    - Filled missing values in the `External Rating` field with zeros.
    - Filled missing values in the `Education level` field with the text "Середня".
    - Added industry ratings data from the `industries.csv` file to the applications DataFrame.

2. **Rating Calculation**:
    - The rating is the sum of scores for the application across 6 criteria and must be a number from 0 to 100.
    - The rating is zero if the `Amount` value is missing or `External Rating` equals zero.
    - The rating is composed of the following components:
        - 20 points if the applicant's age is between 35 and 55.
        - 20 points if the application was submitted on a weekday.
        - 20 points if the applicant is married.
        - 10 points if the applicant is located in Kyiv or the Kyiv region.
        - `Score` value from the `industries.csv` table (ranging from 0 to 20 points).
        - 20 points if `External Rating` is greater than or equal to 7.
        - 20 points subtracted if `External Rating` is less than or equal to 2.

3. **Application Acceptance**:
    - Applications are considered accepted if the rating is greater than zero.

4. **Data Grouping and Visualization**:
    - Grouped the data from the resulting table by the submission week.
    - Plotted the average rating of accepted applications for each week on a graph.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.dates as mdates

# Load data
applications = pd.read_csv('applications.csv')
industries = pd.read_csv('industries.csv')

# Check data structure
applications.info()

# Remove duplicates
applications.drop_duplicates('applicant_id', inplace=True)

# Fill missing values
applications['External Rating'].fillna(0, inplace=True)
applications['Education level'].fillna('Середня', inplace=True)

# Merge datasets
full_df = pd.merge(applications, industries, on='Industry', how='left')

# Calculate the application rating
full_df['Applied at'] = pd.to_datetime(full_df['Applied at'], format='mixed')
full_df['Age Score'] = full_df['Age'].apply(lambda x: 20 if 35 <= x <= 55 else 0)

# Create weekdays list and get day names
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
full_df['Applied Day'] = full_df['Applied at'].apply(lambda x: x.day_name())
full_df['Apply Score'] = full_df['Applied Day'].isin(weekdays) * 20

# Calculate scores
full_df['Marital Score'] = full_df['Marital status'].apply(lambda x: 20 if x == 'Married' else 0)
full_df['Location Score'] = full_df['Location'].apply(lambda x: 10 if x == 'Київ чи область' else 0)
full_df['External Score'] = full_df['External Rating'].apply(lambda x: 20 if x >= 7 else (-20 if x <= 2 else 0)).astype(int)
full_df['Total Score'] = full_df[['Score', 'Apply Score', 'Age Score', 'Marital Score', 'Location Score', 'External Score']].sum(axis=1)
full_df['Total Score'] = full_df['Total Score'].apply(lambda x: 0 if x < 0 else (100 if x > 100 else x))
full_df.loc[full_df['External Rating'] == 0, ['Total Score']] = 0
full_df.loc[full_df['Amount'].isnull(), ['Total Score']] = 0

# Resulting table with accepted applications
full_df.drop(columns=['Score', 'Age Score', 'Apply Score', 'Marital Score', 'Location Score', 'External Score', 'Applied Day'], inplace=True)
accepted = full_df.loc[full_df['Total Score'] > 0, :]

# Group the data from the resulting table by the submission week, and plot the average rating of accepted applications for each week on a graph
accepted.loc[:, 'Application Week'] = accepted['Applied at'].apply(lambda x: x.isocalendar().week)
accepted.loc[:, 'Week Start'] = accepted['Applied at'].dt.to_period('W').apply(lambda y: y.start_time)

# Average rating grouped by submission week
data = round(accepted.groupby('Week Start')['Total Score'].mean().reset_index(), 2)

# Bar chart
data['Week Start'] = data['Week Start'].astype(str)
plt.rcParams['figure.figsize'] = [30, 10]
sns.set(style='whitegrid')
ax = sns.barplot(x='Week Start', y='Total Score', data=data)
plt.title('Average Rating by Application Week', fontdict={'fontsize': 30, 'fontweight': 'bold'}, pad=20)
plt.xlabel(None, fontsize='xx-large')
plt.ylabel('AVG Score', fontsize='xx-large')

# Display values on each bar
for p in ax.patches:
    ax.annotate(f'{(p.get_height()):.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=16, color='black', xytext=(0, 14), textcoords='offset points')
ax.tick_params(axis='both', which='major', labelsize=16)
plt.show()
