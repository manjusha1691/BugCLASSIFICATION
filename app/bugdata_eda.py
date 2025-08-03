# -*- coding: utf-8 -*-


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

"""**Calculate each issue resolution time and group by year**"""

# Commented out IPython magic to ensure Python compatibility.
# %run '/content/drive/MyDrive/Data_Science/BERT/clean_data.ipynb'

"""Plot  to analyse the average issue resolution time per year"""
def calculateAvgIssuePeryear(data):
  # Time taken to resolve an issue
  data['Resolution_Time'] = (data['Resolved'] - data['Created']).dt.days
  # Year in which issue is created
  data['Year'] = data['Created'].dt.year
   # Group by year and aggregate
  yearly_resolution = data.groupby('Year')['Resolution_Time'].agg(pd.Series.mean).reset_index()
  plt.figure(figsize=(10,6))
  sns.barplot(data=yearly_resolution, x='Year', y='Resolution_Time')
  plt.title("Average Time taken to resolve Bugs in a year")
  plt.ylabel('Resolution Time (days)')
  plt.xlabel('Year')
  plt.grid(axis='y')
  plt.show()

  return yearly_resolution



"""Plot  to show issues created each year with different priorities"""

def issues_per_year_priority_linechart(data):
    # Extract year from 'Created' column
    data['Year'] = data['Created'].dt.year

    # Group by Year and Priority, and count number of issues
    issue_counts = data.groupby(['Year', 'Priority']).size().reset_index(name='IssueCount')

    # Pivot for line plot (rows = Year, columns = Priority, values = IssueCount)
    pivot_df = issue_counts.pivot(index='Year', columns='Priority', values='IssueCount').fillna(0)

    # Plot line chart
    plt.figure(figsize=(12, 6))
    pivot_df.plot(marker='o', linewidth=2)
    plt.title('Number of Issues Raised per Year by Priority')
    plt.xlabel('Year')
    plt.ylabel('Number of Issues')
    plt.grid(True)
    plt.legend(title='Priority')
    plt.xticks(pivot_df.index, rotation=45)
    plt.tight_layout()
    plt.show()

    return pivot_df

"""Plot  to show issues Resolved each year with different priorities"""

def resolved_issues_by_priority(data):
    # Ensure 'Resolved' is datetime and not null
    data['Resolved'] = pd.to_datetime(data['Resolved'], errors='coerce')
    resolved_data = data.dropna(subset=['Resolved'])

    # Group by Priority and count resolved issues
    resolved_counts = resolved_data['Priority'].value_counts().reset_index()
    resolved_counts.columns = ['Priority', 'ResolvedCount']

    # Bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=resolved_counts, x='Priority', y='ResolvedCount', palette='Set2')
    plt.title('Number of Resolved Issues by Priority')
    plt.xlabel('Priority')
    plt.ylabel('Resolved Issue Count')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    return resolved_counts

"""PLot to show number of issues created each  year"""
def issues_per_year(data):
  data['Year'] = data['Created'].dt.year
  issues_per_year = data['Year'].value_counts()
  plt.figure(figsize=(10,6))
  sns.barplot(x=issues_per_year.index, y=issues_per_year.values, palette='Set2')
  plt.title('Number of Issues Created each yeat')
  plt.xlabel('Year')
  plt.ylabel('Number of Issues')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

  return issues_per_year

def issues_per_year_perPriority(data):
  data['Year'] = data['Created'].dt.year
  issues_per_year = data.groupby(['Year', 'Priority']).size().reset_index(name='IssueCount')
  plt.figure(figsize=(10,6))
  sns.barplot(x=issues_per_year['Year'], y=issues_per_year['IssueCount'], hue=issues_per_year['Priority'])
  plt.title('Number of Issues per Year by Priority')
  plt.xlabel('Year')
  plt.ylabel('Number of Issues')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()

  return issues_per_year

"""Plot to show number of issues created vs resolved each year"""

def priority_vs_resolved(data):
    # Ensure 'Resolved' is datetime
    data['Resolved'] = pd.to_datetime(data['Resolved'], errors='coerce')

    # Filter rows where 'Resolved' is not null (i.e., the issue is resolved)
    resolved_data = data[data['Resolved'].notnull()]

    # Count resolved issues per priority
    resolved_counts = resolved_data['Priority'].value_counts().reset_index()
    resolved_counts.columns = ['Priority', 'ResolvedCount']

    # Plot
    plt.figure(figsize=(8,5))
    sns.barplot(x='Priority', y='ResolvedCount', data=resolved_counts, palette='Set2')
    plt.title('Resolved Issues by Priority')
    plt.xlabel('Priority')
    plt.ylabel('Number of Resolved Issues')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    return resolved_counts