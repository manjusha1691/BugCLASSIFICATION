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
  # Extracts Year in which issue is created
  data['Year'] = data['Created'].dt.year
   # Group by year and aggregate
  yearly_resolution = data.groupby('Year')['Resolution_Time'].agg(pd.Series.mean).reset_index()

  # Create a bar plot to visualize the average resolution time per year
  fig, ax = plt.subplots(figsize=(10,6))
  sns.barplot(data=yearly_resolution, x='Year', y='Resolution_Time', ax=ax)

  # Set plot title and axis labels
  ax.set_title("Average Time taken to resolve Bugs in a year")
  ax.set_ylabel('Resolution Time (days)')
  ax.set_xlabel('Year')
  ax.grid(axis='y')

  return fig



"""Plot  to show issues created each year with different priorities"""

def issues_per_year_priority_linechart(data):
    
    # Extract year from 'Created' column
    data['Year'] = data['Created'].dt.year

    # Group by Year and Priority, and count number of issues
    issue_counts = data.groupby(['Year', 'Priority']).size().reset_index(name='IssueCount')

    # Pivot for line plot (rows = Year, columns = Priority, values = IssueCount)
    pivot_df = issue_counts.pivot(index='Year', columns='Priority', values='IssueCount').fillna(0)

     # Plot line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_df.plot(ax=ax)

    ax.set_title('Number of Issues Raised per Year by Priority')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Issues')
    ax.grid(True)
    ax.legend(title='Priority')
    ax.set_xticks(pivot_df.index)
    ax.set_xticklabels(pivot_df.index, rotation=45)
    # Optionally return the pivot table if needed for inspection
    return fig

"""Plot  to show issues Resolved each year with different priorities"""

def resolved_issues_by_priority(data):
    # Ensure 'Resolved' is datetime and not null
    data['Resolved'] = pd.to_datetime(data['Resolved'], errors='coerce')
    resolved_data = data.dropna(subset=['Resolved'])

    # Group by Priority and count resolved issues
    resolved_counts = resolved_data['Priority'].value_counts().reset_index()
    resolved_counts.columns = ['Priority', 'ResolvedCount']

    # Bar plot
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=resolved_counts, x='Priority', y='ResolvedCount', palette='Set2', ax = ax)
    ax.set_title('Number of Resolved Issues by Priority')
    ax.set_xlabel('Priority')
    ax.set_ylabel('Resolved Issue Count')
    ax.grid(axis='y')
    plt.tight_layout()

    return fig

"""PLot to show number of issues created each  year"""
def issues_per_year(data):
  #Year in which bug is created
  data['Year'] = data['Created'].dt.year
  #Issues created each year
  issues_per_year = data['Year'].value_counts()
  #plot the graph
  fig, ax = plt.subplots(figsize=(10,6))
  sns.barplot(x=issues_per_year.index, y=issues_per_year.values, palette='Set2', ax = ax)
  ax.set_title('Number of Issues Created each yeat')
  ax.set_xlabel('Year')
  ax.set_ylabel('Number of Issues')
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
  plt.tight_layout()

  return fig

def issues_per_year_perPriority(data):
  data['Year'] = data['Created'].dt.year
  issues_per_year = data.groupby(['Year', 'Priority']).size().reset_index(name='IssueCount')

  #plot the graph
  fig, ax = plt.subplots(figsize=(10,6))
  plt.figure(figsize=(10,6))
  sns.barplot(x=issues_per_year['Year'], y=issues_per_year['IssueCount'], hue=issues_per_year['Priority'], ax = ax)
  ax.set_title('Number of Issues per Year by Priority')
  ax.set_xlabel('Year')
  ax.set_ylabel('Number of Issues')
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

  plt.tight_layout()


  return fig

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
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x='Priority', y='ResolvedCount', data=resolved_counts, palette='Set2', ax = ax)
   
    ax.set_title('Resolved Issues by Priority')
    ax.set_xlabel('Priority')
    ax.set_ylabel('Number of Resolved Issues')
    ax.grid(axis='y')
    plt.tight_layout()
 

    return fig