# -*- coding: utf-8 -*-
"""Analyses VF

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import scipy

#importer le dataset VF

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

df= pd.read_excel('/content/drive/MyDrive/Scibids/Tableau_nettoyeVF.xlsx')

"""# Cleaning"""

print(df.dtypes)

# Drop the 'Unnamed: 0' column from the df DataFrame
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Confirming the column has been dropped
remaining_columns_after_drop = df.columns.tolist()
remaining_columns_after_drop

df['Clients Characteristics Company ID'] = df['Clients Characteristics Company ID'].astype('Int64')
# fix float issue

print(df.dtypes)

"""## Final df"""

# Convert specified columns to float
columns_to_convert = ['CPC', 'CPC*distinct order ID ', 'CPM', 'CPM*distinct order ID ']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

print(df.dtypes)

df['Clients Characteristics Scibids Region'] = df['Clients Characteristics Scibids Region'].replace('US', 'North America')

# Check the data types and missing values
data_info = df.info()

# Adjust the column names in our list
cols_to_clean = ['CPC', 'CPC*distinct order ID ', 'CPM', 'CPM*distinct order ID ']

# Replace '#DIV/0!' with NaN for relevant columns
for col in cols_to_clean:
    df[col] = df[col].replace('#DIV/0!', float('nan')).astype(float)

# Check again for missing values
missing_values = df.isnull().sum()

missing_values

"""## Scibids activity df"""

# Shorten the code when looking for Scibids active data

scibids_active = df[df['Performance Measures Billing Scibids Activity'] == 'Scibids Active']
without_scibids = df[df['Performance Measures Billing Scibids Activity'] == 'Without Scibids']

"""# Client overview, KPI and DSP distribution

## Client overview
"""

# Count the unique values in the "Clients Characteristics Company ID" column
total_clients = scibids_active['Clients Characteristics Company ID'].nunique()

total_clients

# Group by typology and region and calculate the number of unique clients
grouped_typology_region = scibids_active.groupby(['Clients Characteristics Typology', 'Clients Characteristics Scibids Region'])['Clients Characteristics Company ID'].nunique()

# Pivot the grouped data to get regions as columns and typologies as rows
pivot_typology_region = grouped_typology_region.unstack('Clients Characteristics Scibids Region')

# Convert values to integers
pivot_typology_region = pivot_typology_region.fillna(0).astype(int)

pivot_typology_region

# Create a mask for values that are 0
mask = pivot_typology_region == 0

# Setting up the figure and axis
plt.figure(figsize=(12, 8))

# Creating the heatmap with the mask
sns.heatmap(pivot_typology_region, annot=True, cmap="YlGnBu", fmt="d", cbar_kws={'label': 'Total Clients'}, mask=mask)

# Setting title and labels
plt.title("Number of Unique Clients by Typology and Region")
plt.xlabel("Region")
plt.ylabel("Typology")
plt.tight_layout()

# Display the heatmap
plt.show()

# Set up the figure and axis
plt.figure(figsize=(15, 8))

# Width of each individual bar
bar_width = 0.15

# Set up the x positions for groups
r = np.arange(len(pivot_typology_region.index))

# Create bars for each region
for idx, column in enumerate(pivot_typology_region.columns):
    plt.bar(r + idx * bar_width, pivot_typology_region[column], width=bar_width, label=column)

# Configure the layout
plt.xlabel('Typology', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(pivot_typology_region.index))], pivot_typology_region.index, rotation=45)
plt.ylabel('Total Clients')
plt.title('Number of Unique Clients by Typology and Region')
plt.legend(title='Region', loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.tight_layout()
plt.show()

"""##  Evolution of number of clients over time"""

# Group by month, typology, and region and count unique company IDs
client_evolution = scibids_active.groupby(['Performance Measures Day Tz Month',
                                 'Clients Characteristics Typology',
                                 'Clients Characteristics Scibids Region'])['Clients Characteristics Company ID'].nunique().reset_index()

client_evolution

# Group data by month and region, then count unique clients
clients_over_time_region = scibids_active.groupby(['Performance Measures Day Tz Month', 'Clients Characteristics Scibids Region'])['Clients Characteristics Company ID'].nunique().unstack()

clients_over_time_region

# Plot the evolution of number of clients over time for each region
plt.figure(figsize=(15, 7))
clients_over_time_region.plot(ax=plt.gca())
plt.title('Evolution of Number of Clients Over Time by Region')
plt.xlabel('Month')
plt.ylabel('Number of Unique Clients')
plt.grid(True)
plt.legend(title='Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""## KPI distribution"""

# KPI preference
kpi_by_typology = scibids_active.groupby(['Clients Characteristics Typology', 'unified_KPI'])['Insertion Orders Distinct Count of IOs'].sum().unstack()
kpi_by_typology

# Convert the sums into percentages
kpi_distribution_typology = kpi_by_typology.divide(kpi_by_typology.sum(axis=1), axis=0) * 100

kpi_distribution_typology

# Create a mask for values that are 0
mask = kpi_distribution_typology == 0

# Plotting the heatmap with the mask
plt.figure(figsize=(16, 8))
sns.heatmap(kpi_distribution_typology, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Percentage'}, mask=mask)
plt.title('KPI Distribution by Client Typology (Based on Distinct Insertion Orders) %')
plt.xlabel('KPI')
plt.ylabel('Client Typology')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

kpi_by_region = scibids_active.groupby(['Clients Characteristics Scibids Region', 'unified_KPI'])['Insertion Orders Distinct Count of IOs'].sum().unstack()
kpi_by_region

# Convert the sums into percentages
kpi_distribution_region = kpi_by_region.divide(kpi_by_region.sum(axis=1), axis=0) * 100

kpi_distribution_region

# Create a mask for values that are 0
mask = kpi_distribution_region == 0

# Plotting the heatmap with the mask
plt.figure(figsize=(16, 8))
sns.heatmap(kpi_distribution_region, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Percentage'}, mask=mask)
plt.title('KPI Distribution by Region (Based on Distinct Insertion Orders) %')
plt.xlabel('KPI')
plt.ylabel('Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""## Campaigns distribution"""

campaigns_per_month = scibids_active.groupby(['Performance Measures Day Tz Month'])['Insertion Orders Distinct Count of IOs'].sum()
campaigns_per_month

# Group by client typology and region to sum the number of campaigns  without Scibids filtering
campaigns_by_typology_region_all = df.groupby(['Clients Characteristics Typology', 'Clients Characteristics Scibids Region'])['Insertion Orders Distinct Count of IOs'].sum().unstack()

# Plot the sum of campaigns by client typology for each region
plt.figure(figsize=(15, 7))
campaigns_by_typology_region_all.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Number of Campaigns by Client Typology and Region (All Data)')
plt.xlabel('Client Typology')
plt.ylabel('Number of Campaigns')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Filter for rows where Scibids is activated

# Sum the number of campaigns (using "Insertion Orders Distinct Count of IOs") by client typology and region
campaigns_by_typology_region = scibids_active.groupby(['Clients Characteristics Typology', 'Clients Characteristics Scibids Region'])['Insertion Orders Distinct Count of IOs'].sum().unstack()

# Fill NaN values with zeros for visualization
campaigns_by_typology_region_filled = campaigns_by_typology_region.fillna(0)

# Plotting the sum of campaigns by client typology for each region
plt.figure(figsize=(15, 7))
campaigns_by_typology_region_filled.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Number of Campaigns by Client Typology and Region (With Scibids Activation)')
plt.xlabel('Client Typology')
plt.ylabel('Number of Campaigns')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

"""## DSP distribution"""

# DSP preference by Region
dsp_by_typology = scibids_active.groupby(['Clients Characteristics Typology', 'Accessible IDs Dsp'])['Insertion Orders Distinct Count of IOs'].sum().unstack()
dsp_by_typology

# Normalize the values
dsp_distribution_typology = dsp_by_typology.divide(dsp_by_typology.sum(axis=1), axis=0)
dsp_distribution_typology

# Visualise

plt.figure(figsize=(15, 10))
sns.heatmap(dsp_distribution_typology, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Percentage of DSP Usage'}, fmt=".2%")
plt.title('Distribution of DSPs by Typology')
plt.xlabel('Demand-Side Platforms (DSPs)')
plt.ylabel('Typology')
plt.show()

# DSP preference by Region
dsp_by_region = scibids_active.groupby(['Clients Characteristics Scibids Region', 'Accessible IDs Dsp'])['Insertion Orders Distinct Count of IOs'].sum().unstack()
dsp_by_region

# Normalize the values
dsp_distribution_region = dsp_by_region.divide(dsp_by_region.sum(axis=1), axis=0)
dsp_distribution_region

# Visualise

plt.figure(figsize=(15, 10))
sns.heatmap(dsp_distribution_region, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Percentage of DSP Usage'}, fmt=".2%")
plt.title('Distribution of DSPs by Region')
plt.xlabel('Demand-Side Platforms (DSPs)')
plt.ylabel('Region')
plt.show()

# Grouping by Region and Sub-DSP to get the sum of distinct insertion orders for each Sub-DSP per region
sub_dsp_region_preference = scibids_active.groupby(['Clients Characteristics Scibids Region', 'Accessible IDs Sub Dsp'])['Insertion Orders Distinct Count of IOs'].sum().unstack().fillna(0)

# Sorting the values for better visualization
sub_dsp_region_preference = sub_dsp_region_preference[sub_dsp_region_preference.sum().sort_values(ascending=False).index]

sub_dsp_region_preference

# Grouping by Region and Sub-DSP to get the sum of distinct insertion orders for each Sub-DSP per region
sub_dsp_typology_preference = scibids_active.groupby(['Clients Characteristics Typology', 'Accessible IDs Sub Dsp'])['Insertion Orders Distinct Count of IOs'].sum().unstack().fillna(0)

# Sorting the values for better visualization
sub_dsp_typology_preference = sub_dsp_typology_preference[sub_dsp_typology_preference.sum().sort_values(ascending=False).index]
sub_dsp_typology_preference

# Grouping by Region, Typology, and Sub-DSP to get the sum of distinct insertion orders
sub_dsp = scibids_active.groupby(['Clients Characteristics Scibids Region', 'Clients Characteristics Typology', 'Accessible IDs Sub Dsp'])['Insertion Orders Distinct Count of IOs'].sum().reset_index()

# Pivot the table to move the Sub-DSP values to columns
sub_dsp_preference = sub_dsp.pivot_table(index=['Clients Characteristics Scibids Region', 'Clients Characteristics Typology'],
                                        columns='Accessible IDs Sub Dsp',
                                        values='Insertion Orders Distinct Count of IOs',
                                        aggfunc='sum')

# Reorder the columns to have "Display" and "Trueview" as top-level columns
column_order = [col for col in ['Display', 'TrueView'] if col in pivoted_data.columns]
sub_dsp_preference = sub_dsp_preference[column_order]

# Convert values to integers
sub_dsp_preference = sub_dsp_preference.fillna(0).astype(int)


sub_dsp_preference

# Group by both 'Clients Characteristics Scibids Region' and 'Clients Characteristics Typology',
# along with 'Accessible IDs Sub Dsp', to get the sum of distinct insertion orders for each combination
combined_pivot = (scibids_active
                  .groupby(['Clients Characteristics Scibids Region',
                            'Clients Characteristics Typology',
                            'Accessible IDs Sub Dsp'])
                  ['Insertion Orders Distinct Count of IOs'].sum().unstack())

# Fill NaN values with 0 for better visualization
combined_pivot.fillna(0, inplace=True)

combined_pivot

combined_pivot.astype(int).style.background_gradient(cmap='YlGnBu')

# Plotting the DSP preferences per region
dsp_by_region.plot(kind='bar', figsize=(15, 8), stacked=True, colormap='tab20c')

plt.title('DSP Preferences by Region based on Distinct Insertion Orders')
plt.ylabel('Sum of Distinct Insertion Orders')
plt.xlabel('Region')
plt.xticks(rotation=45)
plt.legend(title='DSPs', loc="upper left", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()

# Plotting the Sub-DSP preferences per region
sub_dsp_region_preference.plot(kind='bar', figsize=(15, 8), stacked=True, colormap='tab10')

plt.title('Sub-DSP Preferences by Region based on Distinct Insertion Orders')
plt.ylabel('Sum of Distinct Insertion Orders')
plt.xlabel('Region')
plt.xticks(rotation=45)
plt.legend(title='Sub-DSPs', loc="upper left", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()

"""## Insertion orders distribution by KPI and DSP"""

# Aggregating data by DSP and KPI based on 'Insertion Orders Distinct Count of IOs'
kpi_dsp_aggregation = (scibids_active
                       .groupby(['Accessible IDs Dsp', 'unified_KPI'])
                       ['Insertion Orders Distinct Count of IOs'].sum().unstack())

# Filling NaN values with 0 for better visualization
kpi_dsp_aggregation.fillna(0, inplace=True)
kpi_dsp_aggregation

# Plotting heatmap for KPIs by DSP based on 'Insertion Orders Distinct Count of IOs'


# Modifying the heatmap code to not show values that are 0

plt.figure(figsize=(14, 8))
sns.heatmap(kpi_dsp_aggregation, cmap='YlGnBu', annot=True, fmt=".0f", linewidths=.5,
            cbar_kws={'label': 'Number of Campaigns'},
            mask=kpi_dsp_aggregation == 0)  # Using a mask to hide 0 values
plt.title('Number of Campaigns by KPI and DSP')
plt.xlabel('unified_KPI')
plt.ylabel('DSP')
plt.show()

# Aggregating data by Region, DSP, and KPI based on 'Insertion Orders Distinct Count of IOs'
kpi_dsp_region_aggregation = (scibids_active
                       .groupby(['Clients Characteristics Scibids Region', 'Accessible IDs Dsp', 'Insertion Orders Kpi to Optimize'])
                       ['Insertion Orders Distinct Count of IOs'].sum().unstack())

# Filling NaN values with 0 for better visualization
kpi_dsp_region_aggregation.fillna(0, inplace=True)
kpi_dsp_region_aggregation

# Grouping by Sub DSP to see the distribution of campaigns
campaigns_sub_dsp = scibids_active.groupby('Accessible IDs Sub Dsp')['Insertion Orders Distinct Count of IOs'].sum()

# Plotting
plt.figure(figsize=(14, 7))
campaigns_sub_dsp.sort_values(ascending=False).plot(kind='bar', color='teal')

plt.title('Distribution of Campaigns across Sub DSPs')
plt.xlabel('Sub DSP')
plt.ylabel('Count of Campaigns')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""# Revenue analysis

## Revenue by typology and insertion orders count
"""

# Grouping by Typology and aggregating the number of campaigns and revenue
typology_aggregated = scibids_active.groupby('Clients Characteristics Typology').agg({
    'Insertion Orders Distinct Count of IOs': 'sum',
    'Performance Measures Revenue USD': 'sum'
}).reset_index()

# Sorting by Revenue for better visualization
typology_aggregated = typology_aggregated.sort_values(by='Performance Measures Revenue USD', ascending=False)

typology_aggregated

revenue_per_month = scibids_active.groupby(['Performance Measures Day Tz Month', 'Clients Characteristics Typology', 'Insertion Orders Distinct Count of IOs'])['Performance Measures Revenue USD'].sum()
revenue_per_month

"""## Total & Normalised Revenue Distribution per DSP by Typology"""

## Total


# Group by both Typology and DSP to calculate the revenue for each combination
revenue_distribution_dsp_typology = scibids_active.groupby(['Clients Characteristics Typology', 'Accessible IDs Dsp'])['Performance Measures Revenue USD'].sum().reset_index()

# Calculate the total revenue per typology for normalization
total_revenue_per_typology = revenue_distribution_dsp_typology.groupby('Clients Characteristics Typology')['Performance Measures Revenue USD'].transform('sum')

# Calculate the percentage of total revenue for each DSP within each typology
revenue_distribution_dsp_typology['% of Total Revenue per Typology'] = (revenue_distribution_dsp_typology['Performance Measures Revenue USD'] / total_revenue_per_typology) * 100

# Sort the data by typology and then by percentage revenue in descending order
revenue_distribution_dsp_typology = revenue_distribution_dsp_typology.sort_values(by=['Clients Characteristics Typology', '% of Total Revenue per Typology'], ascending=[True, False])

revenue_distribution_dsp_typology

# Grouping data by client typology and DSP, then summing the revenue
revenue_by_typology_dsp = scibids_active.groupby(['Clients Characteristics Typology', 'Accessible IDs Dsp'])['Performance Measures Revenue USD'].sum().unstack().fillna(0)

# Visualizing the revenue distribution across DSPs for each client typology
revenue_by_typology_dsp.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
plt.title('Revenue Distribution across DSPs by Client Typology')
plt.xlabel('Client Typology')
plt.ylabel('Total Revenue (USD)')
plt.legend(title='DSP', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Group by Sub DSPs and sum the revenue
sub_dsp_revenue = scibids_active.groupby('Accessible IDs Sub Dsp')['Performance Measures Revenue USD'].sum()

# Sorting the values for better visualization
sub_dsp_revenue_sorted = sub_dsp_revenue.sort_values(ascending=False)

# Plotting
plt.figure(figsize=(15, 8))
sub_dsp_revenue_sorted.plot(kind='barh', color='teal')
plt.title('Distribution of Revenue Across Different Sub DSPs')
plt.xlabel('Revenue USD')
plt.ylabel('Sub DSPs')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Heatmap
from matplotlib.colors import ListedColormap

# Pivot the dataframe to get a matrix suitable for heatmap
heatmap_data = revenue_distribution_dsp_typology.pivot('Clients Characteristics Typology', 'Accessible IDs Dsp', '% of Total Revenue per Typology').fillna(0)

# Plotting the heatmap
mask = heatmap_data == 0

# Custom annotation function
def custom_annotation(val):
    return '' if val == 0 else f'{val:.2f}'

# Create an array of annotations
annotations = np.vectorize(custom_annotation)(heatmap_data.values)

# Create a custom colormap. Colors below 1e-10 will be white, and the rest will follow 'YlGnBu'
cmap = ListedColormap(['white'] + sns.color_palette("YlGnBu", 100).as_hex())

# Plotting the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=annotations, cmap=cmap, fmt="", linewidths=.5, mask=mask)
plt.title('Percentage of Total Revenue per DSP within each Typology')
plt.xlabel('DSP')
plt.ylabel('Client Typology')
plt.tight_layout()
plt.show()

## Per insertion order

# Group by both Typology and DSP to calculate the normalized revenue for each combination
normalised_revenue_distribution_dsp_typology = scibids_active.groupby(['Clients Characteristics Typology', 'Accessible IDs Dsp'])\
    .apply(lambda x: x['Performance Measures Revenue USD'].sum() / x['Insertion Orders Distinct Count of IOs'].sum()).reset_index()

normalised_revenue_distribution_dsp_typology.columns = ['Clients Characteristics Typology', 'Accessible IDs Dsp', 'Normalized Revenue']

# Calculate the total normalized revenue per typology for normalization
total_normalized_revenue_per_typology = normalised_revenue_distribution_dsp_typology.groupby('Clients Characteristics Typology')['Normalized Revenue'].transform('sum')

# Calculate the percentage of total normalized revenue for each DSP within each typology
normalised_revenue_distribution_dsp_typology['% of Total Normalized Revenue per Typology'] = (normalised_revenue_distribution_dsp_typology['Normalized Revenue'] / total_normalized_revenue_per_typology) * 100

normalised_revenue_distribution_dsp_typology

# Pivot the dataframe to get a matrix suitable for heatmap
heatmap_data_dsp = normalised_revenue_distribution_dsp_typology.pivot('Clients Characteristics Typology', 'Accessible IDs Dsp', '% of Total Normalized Revenue per Typology').fillna(0)

# Create a mask for 0 values
mask = heatmap_data_dsp == 0

# Custom annotation function
def custom_annotation(val):
    return '' if val == 0 else f'{val:.2f}'

# Create an array of annotations
annotations = np.vectorize(custom_annotation)(heatmap_data_dsp.values)

# Create a custom colormap. Colors below 1e-10 will be white, and the rest will follow 'YlGnBu'
cmap = ListedColormap(['white'] + sns.color_palette("YlGnBu", 100).as_hex())

# Plotting the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data_dsp, annot=annotations, cmap=cmap, fmt="", linewidths=.5, mask=mask)
plt.title('Percentage of Total Normalized Revenue per DSP within each Typology')
plt.xlabel('DSP')
plt.ylabel('Client Typology')
plt.tight_layout()
plt.show()

"""## Total & Normalised Revenue Distribution per DSP by Region"""

## Total

# Group by both Region and DSP to calculate the revenue for each combination
revenue_distribution_dsp_region = scibids_active.groupby(['Clients Characteristics Scibids Region', 'Accessible IDs Dsp'])['Performance Measures Revenue USD'].sum().reset_index()

# Calculate the total revenue per region for normalization
total_revenue_per_region = revenue_distribution_dsp_region.groupby('Clients Characteristics Scibids Region')['Performance Measures Revenue USD'].transform('sum')

# Calculate the percentage of total revenue for each DSP within each region
revenue_distribution_dsp_region['% of Total Revenue per Region'] = (revenue_distribution_dsp_region['Performance Measures Revenue USD'] / total_revenue_per_region) * 100

revenue_distribution_dsp_region

# Pivot the dataframe to get a matrix suitable for heatmap
heatmap_data_region = revenue_distribution_dsp_region.pivot('Clients Characteristics Scibids Region', 'Accessible IDs Dsp', '% of Total Revenue per Region').fillna(0)

# Create a mask for 0 values
mask = heatmap_data_region == 0

# Custom annotation function
def custom_annotation(val):
    return '' if val == 0 else f'{val:.2f}'

# Create an array of annotations
annotations = np.vectorize(custom_annotation)(heatmap_data_region.values)

# Create a custom colormap. Colors below 1e-10 will be white, and the rest will follow 'YlGnBu'
cmap = ListedColormap(['white'] + sns.color_palette("YlGnBu", 100).as_hex())

# Plotting the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data_region, annot=annotations, cmap=cmap, fmt="", linewidths=.5, vmin=1, vmax=100, mask=mask)
plt.title('Percentage of Total Revenue per DSP within each Region')
plt.xlabel('DSP')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

## Per insertion order

# Group by both Region and DSP to calculate the normalized revenue for each combination
normalized_revenue_distribution_dsp_region = scibids_active.groupby(['Clients Characteristics Scibids Region', 'Accessible IDs Dsp'])\
    .apply(lambda x: x['Performance Measures Revenue USD'].sum() / x['Insertion Orders Distinct Count of IOs'].sum()).reset_index()

normalized_revenue_distribution_dsp_region.columns = ['Clients Characteristics Scibids Region', 'Accessible IDs Dsp', 'Normalized Revenue']

# Calculate the total normalized revenue per region for normalization
total_normalized_revenue_per_region = normalized_revenue_distribution_dsp_region.groupby('Clients Characteristics Scibids Region')['Normalized Revenue'].transform('sum')

# Calculate the percentage of total normalized revenue for each DSP within each region
normalized_revenue_distribution_dsp_region['% of Total Normalized Revenue per Region'] = (normalized_revenue_distribution_dsp_region['Normalized Revenue'] / total_normalized_revenue_per_region) * 100

normalized_revenue_distribution_dsp_region

# Pivot the dataframe to get a matrix suitable for heatmap
heatmap_data_normalized_region = normalized_revenue_distribution_dsp_region.pivot('Clients Characteristics Scibids Region', 'Accessible IDs Dsp', '% of Total Normalized Revenue per Region').fillna(0)


# Create a mask for 0 values
mask = heatmap_data_normalized_region == 0

# Custom annotation function
def custom_annotation(val):
    return '' if val == 0 else f'{val:.2f}'

# Create an array of annotations
annotations = np.vectorize(custom_annotation)(heatmap_data_normalized_region.values)

# Create a custom colormap. Colors below 1e-10 will be white, and the rest will follow 'YlGnBu'
cmap = ListedColormap(['white'] + sns.color_palette("YlGnBu", 100).as_hex())

# Plotting the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data_normalized_region, annot=annotations, cmap=cmap, fmt="", linewidths=.5, vmin=1, vmax=100, mask=mask)
plt.title('Percentage of Total Normalized Revenue per DSP within each Region')
plt.xlabel('DSP')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

"""## Total & Normalised Revenue Distribution per KPI by Typology"""

## Total

# Group by both Typology and KPI to calculate the revenue for each combination
revenue_distribution_kpi_typology = scibids_active.groupby(['Clients Characteristics Typology', 'unified_KPI'])['Performance Measures Revenue USD'].sum().reset_index()

# Calculate the total revenue per typology for normalization
total_revenue_per_typology_kpi = revenue_distribution_kpi_typology.groupby('Clients Characteristics Typology')['Performance Measures Revenue USD'].transform('sum')

# Calculate the percentage of total revenue for each KPI within each typology
revenue_distribution_kpi_typology['% of Total Revenue per Typology'] = (revenue_distribution_kpi_typology['Performance Measures Revenue USD'] / total_revenue_per_typology_kpi) * 100

# Pivot the dataframe to get a matrix suitable for heatmap
heatmap_data_typology_kpi = revenue_distribution_kpi_typology.pivot('Clients Characteristics Typology', 'unified_KPI', '% of Total Revenue per Typology').fillna(0)

# Create a mask for 0 values
mask = heatmap_data_typology_kpi == 0

from matplotlib.colors import ListedColormap

# Custom annotation function
def custom_annotation(val):
    return '' if val == 0 else f'{val:.2f}'

# Create an array of annotations
annotations = np.vectorize(custom_annotation)(heatmap_data_typology_kpi.values)

# Create a custom colormap. Colors below 1e-10 will be white, and the rest will follow 'YlGnBu'
cmap = ListedColormap(['white'] + sns.color_palette("YlGnBu", 100).as_hex())

# Plotting the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data_typology_kpi, annot=annotations, cmap=cmap, fmt="", linewidths=.5, vmin=1e-10, vmax=82, mask=mask)
plt.title('Percentage of Total Revenue per KPI within each Typology')
plt.xlabel('KPI')
plt.ylabel('Typology')
plt.tight_layout()
plt.show()

## Per insertion order

# Group by both Typology and KPI to calculate the normalized revenue for each combination
normalized_revenue_distribution_kpi_typology = scibids_active.groupby(['Clients Characteristics Typology', 'unified_KPI'])\
    .apply(lambda x: x['Performance Measures Revenue USD'].sum() / x['Insertion Orders Distinct Count of IOs'].sum()).reset_index()

normalized_revenue_distribution_kpi_typology.columns = ['Clients Characteristics Typology', 'unified_KPI', 'Normalized Revenue']

# Calculate the total normalized revenue per typology for normalization
total_normalized_revenue_per_typology_kpi = normalized_revenue_distribution_kpi_typology.groupby('Clients Characteristics Typology')['Normalized Revenue'].transform('sum')

# Calculate the percentage of total normalized revenue for each KPI within each typology
normalized_revenue_distribution_kpi_typology['% of Total Normalized Revenue per Typology'] = (normalized_revenue_distribution_kpi_typology['Normalized Revenue'] / total_normalized_revenue_per_typology_kpi) * 100

# Pivot the dataframe to get a matrix suitable for heatmap
heatmap_data_normalized_typology_kpi = normalized_revenue_distribution_kpi_typology.pivot('Clients Characteristics Typology', 'unified_KPI', '% of Total Normalized Revenue per Typology').fillna(0)

# Create a mask for 0 values
mask = heatmap_data_normalized_typology_kpi == 0

# Custom annotation function
def custom_annotation(val):
    return '' if val == 0 else f'{val:.2f}'

# Create an array of annotations
annotations = np.vectorize(custom_annotation)(heatmap_data_normalized_typology_kpi.values)

# Create a custom colormap. Colors below 1e-10 will be white, and the rest will follow 'YlGnBu'
cmap = ListedColormap(['white'] + sns.color_palette("YlGnBu", 100).as_hex())

# Plotting the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data_normalized_typology_kpi, annot=annotations, cmap=cmap, fmt="", linewidths=.5, vmin=1e-10, vmax=82, mask=mask)
plt.title('Percentage of Total Normalized Revenue per KPI within each Typology')
plt.xlabel('KPI')
plt.ylabel('Typology')
plt.tight_layout()
plt.show()

"""## Total & Normalised Revenue Distribution per KPI by Region"""

## Total

# Group by both Region and KPI to calculate the revenue for each combination
revenue_distribution_kpi_region = scibids_active.groupby(['Clients Characteristics Scibids Region', 'unified_KPI'])['Performance Measures Revenue USD'].sum().reset_index()

# Calculate the total revenue per region for normalization
total_revenue_per_region = revenue_distribution_kpi_region.groupby('Clients Characteristics Scibids Region')['Performance Measures Revenue USD'].transform('sum')

# Calculate the percentage of total revenue for each KPI within each region
revenue_distribution_kpi_region['% of Total Revenue per Region'] = (revenue_distribution_kpi_region['Performance Measures Revenue USD'] / total_revenue_per_region) * 100

revenue_distribution_kpi_region

# Pivot the dataframe to get a matrix suitable for heatmap
heatmap_data_kpi_region = revenue_distribution_kpi_region.pivot('Clients Characteristics Scibids Region', 'unified_KPI', '% of Total Revenue per Region').fillna(0)

# Create a mask for 0 values
mask = heatmap_data_kpi_region == 0

# Custom annotation function
def custom_annotation(val):
    return '' if val == 0 else f'{val:.2f}'

# Create an array of annotations
annotations = np.vectorize(custom_annotation)(heatmap_data_kpi_region.values)

# Create a custom colormap. Colors below 1e-10 will be white, and the rest will follow 'YlGnBu'
cmap = ListedColormap(['white'] + sns.color_palette("YlGnBu", 100).as_hex())

# Plotting the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data_kpi_region, annot=annotations, cmap=cmap, fmt="", linewidths=.5, mask=mask)
plt.title('Percentage of Total Revenue per KPI within each Region')
plt.xlabel('KPI')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

## Per insertion order

# Group by both Region and KPI to calculate the revenue for each combination
revenue_distribution_kpi_region = scibids_active.groupby(['Clients Characteristics Scibids Region', 'unified_KPI']).apply(lambda x: x['Performance Measures Revenue USD'].sum() / x['Insertion Orders Distinct Count of IOs'].sum()).reset_index()

# Rename the columns for clarity
revenue_distribution_kpi_region.columns = ['Clients Characteristics Scibids Region', 'unified_KPI', 'Normalized Revenue']

# Calculate the total normalized revenue per region for percentage calculation
total_normalized_revenue_per_region = revenue_distribution_kpi_region.groupby('Clients Characteristics Scibids Region')['Normalized Revenue'].transform('sum')

# Calculate the percentage of total normalized revenue for each KPI within each region
revenue_distribution_kpi_region['% of Total Normalized Revenue per Region'] = (revenue_distribution_kpi_region['Normalized Revenue'] / total_normalized_revenue_per_region) * 100

revenue_distribution_kpi_region

# Pivot the dataframe to get a matrix suitable for heatmap
heatmap_data_kpi_region = revenue_distribution_kpi_region.pivot('Clients Characteristics Scibids Region', 'unified_KPI', '% of Total Normalized Revenue per Region').fillna(0)

# Create a mask for 0 values
mask = heatmap_data_kpi_region == 0

# Custom annotation function
def custom_annotation(val):
    return '' if val == 0 else f'{val:.2f}'

# Create an array of annotations
annotations = np.vectorize(custom_annotation)(heatmap_data_kpi_region.values)

# Create a custom colormap. Colors below 1e-10 will be white, and the rest will follow 'YlGnBu'
cmap = ListedColormap(['white'] + sns.color_palette("YlGnBu", 100).as_hex())

# Plotting the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data_kpi_region, annot=annotations, cmap=cmap, fmt="", linewidths=.5, mask=mask)
plt.title('Percentage of Total Normalized Revenue per KPI within each Region')
plt.xlabel('KPI')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

"""## Top 10 clients and campaigns per revenue"""

# Top 10 clients with highest revenue
top_clients_revenue = scibids_active.groupby('Clients Characteristics Company Name')['Performance Measures Revenue USD'].sum().nlargest(10)


top_clients_revenue

# Top 10 campaigns with highest revenue

top_campaigns_revenue = scibids_active.groupby('Insertion Orders Distinct Count of IOs')['Performance Measures Revenue USD'].sum().nlargest(10)
top_campaigns_revenue

"""## Calculating the average revenue per campaign based on Scibids Activity and Client Typology

"""

# Grouping the data based on Scibids Activity, Client Typology and calculating the average revenue per campaign
avg_revenue_per_campaign_scibids = (df.groupby(['Clients Characteristics Typology', 'Performance Measures Billing Scibids Activity'])
                                    ['Performance Measures Revenue USD', 'Insertion Orders Distinct Count of IOs']
                                    .sum())

avg_revenue_per_campaign_scibids['Avg Revenue per Campaign'] = avg_revenue_per_campaign_scibids['Performance Measures Revenue USD'] / avg_revenue_per_campaign_scibids['Insertion Orders Distinct Count of IOs']
avg_revenue_per_campaign_scibids = avg_revenue_per_campaign_scibids.reset_index()

# Filtering for Scibids Active and Without Scibids
active_scibids = avg_revenue_per_campaign_scibids[avg_revenue_per_campaign_scibids['Performance Measures Billing Scibids Activity'] == 'Scibids Active']
without_scibids = avg_revenue_per_campaign_scibids[avg_revenue_per_campaign_scibids['Performance Measures Billing Scibids Activity'] == 'Without Scibids']

# Plotting
plt.figure(figsize=(14, 7))

barWidth = 0.35
r1 = np.arange(len(active_scibids))
r2 = [x + barWidth for x in r1]

plt.bar(r1, active_scibids['Avg Revenue per Campaign'], color='blue', width=barWidth, edgecolor='white', label='Scibids Active')
plt.bar(r2, without_scibids['Avg Revenue per Campaign'], color='red', width=barWidth, edgecolor='white', label='Without Scibids')

plt.xlabel('Client Typology', fontweight='bold')
plt.ylabel('Average Revenue per Campaign', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(active_scibids))], active_scibids['Clients Characteristics Typology'], rotation=45)
plt.legend()
plt.title('Comparison of Average Revenue per Campaign (Scibids Active vs Without Scibids)')

plt.tight_layout()
plt.show()

"""## Normalised Revenue per KPI and DSP Combination"""

# Calculating the normalised revenue per distinct insertion order for each KPI and DSP combination
norm_revenue_per_order_per_kpi_dsp = (scibids_active
                                     .groupby(['unified_KPI', 'Accessible IDs Dsp'])
                                     .apply(lambda x: x['Performance Measures Revenue USD'].sum() / x['Insertion Orders Distinct Count of IOs'].sum())
                                     .unstack())

# Visualizing the data
plt.figure(figsize=(14, 8))
sns.heatmap(norm_revenue_per_order_per_kpi_dsp, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5)
plt.title('Normalised Revenue per Distinct Insertion Order for Each KPI and DSP Combination')
plt.xlabel('DSP')
plt.ylabel('KPI Type')
plt.tight_layout()
plt.show()

"""Analysis:


## TradeDesk preference distribution - high revenue
"""

# Filtering the data to only include entries where the DSP is TheTradeDesk
tradedesk_data = scibids_active[scibids_active['Accessible IDs Dsp'] == 'TheTradeDesk']

# Grouping by region and typology to count the number of occurrences
tradedesk_region_typology = tradedesk_data.groupby(['Clients Characteristics Scibids Region', 'Clients Characteristics Typology']).size().unstack()

tradedesk_region_typology

# Grouping by region and typology to count the number of occurrences for TheTradeDesk
tradedesk_region_typology = tradedesk_data.groupby(['Clients Characteristics Scibids Region', 'Clients Characteristics Typology']).size().unstack()

tradedesk_region_typology.fillna(0)  # Filling NaN values with 0 for better representation

"""Big 6 Omnicom is a high revenue generator in North America because of Traderdesk

## Top 20 clients based on revenue, insertions orders, grouped by DSP, region, typology
"""

#
grouped_data = scibids_active.groupby(
    ['Clients Characteristics Scibids Region',
     'Clients Characteristics Company Name',
     'Accessible IDs Client Name',
     'Accessible IDs Advertiser Name',
     'Clients Characteristics Typology',
     'Accessible IDs Dsp']
).agg({
    'Performance Measures Revenue USD': 'sum',
    'Insertion Orders Distinct Count of IOs': 'sum'
}).reset_index()

# Sorting the data based on revenue
grouped_data_sorted = grouped_data.sort_values(by='Performance Measures Revenue USD', ascending=False)

# Displaying the top entities
top_entities = grouped_data_sorted.head(20)
top_entities

"""## Top 20 clients total revenue and IOs per DSP"""

# Check how many times each DSP appears in the top 20
dsp_counts = top_entities['Accessible IDs Dsp'].value_counts()

dsp_counts



# Aggregate revenue by DSP in the top 20
revenue_by_dsp = top_entities.groupby('Accessible IDs Dsp')['Performance Measures Revenue USD'].sum()

revenue_by_dsp

# Aggregate the number of campaigns by DSP in the top 20
campaigns_by_dsp = top_entities.groupby('Accessible IDs Dsp')['Insertion Orders Distinct Count of IOs'].sum()
campaigns_by_dsp

# Merge the three DSP-related insights into a single DataFrame
dsp_summary = pd.concat([
    dsp_counts.rename('DSP Counts in Top 20'),
    revenue_by_dsp.rename('Total Revenue by DSP'),
    campaigns_by_dsp.rename('Total Insertion Orders by DSP')
], axis=1).fillna(0)  # Fill any NaN values with 0

# Convert the 'Total Revenue by DSP' column to integers
dsp_summary['Total Revenue by DSP'] = dsp_summary['Total Revenue by DSP'].astype(int)

dsp_summary

# Set up the aesthetics
sns.set_theme()

# Plotting DSP Counts in Top 20
plt.figure(figsize=(12, 6))
sns.barplot(x=dsp_summary.index, y=dsp_summary['DSP Counts in Top 20'])
plt.title('DSP Counts in Top 20')
plt.xticks(rotation=45)
plt.show()

# Plotting Total Revenue by DSP
plt.figure(figsize=(12, 6))
sns.barplot(x=dsp_summary.index, y=dsp_summary['Total Revenue by DSP'])
plt.title('Total Revenue by DSP')
plt.xticks(rotation=45)
plt.show()

# Plotting Total Campaigns by DSP
plt.figure(figsize=(12, 6))
sns.barplot(x=dsp_summary.index, y=dsp_summary['Total Campaigns by DSP'])
plt.title('Total Campaigns by DSP')
plt.xticks(rotation=45)
plt.show()

## Zetaglobal top client per revenue

# examine


zetaglobal_rows = df[df['Clients Characteristics Company Name'] == 'Zetaglobal']

zetaglobal_rows

from google.colab import files
zetaglobal_rows.to_csv('zetaglobal_rows.csv')
files.download('zetaglobal_rows.csv')

# Calculate average CPC and CPM per insertion order again
zetaglobal_rows['CPC_per_IO'] = zetaglobal_rows['CPC'] / zetaglobal_rows['Insertion Orders Distinct Count of IOs']
zetaglobal_rows['CPM_per_IO'] = zetaglobal_rows['CPM'] / zetaglobal_rows['Insertion Orders Distinct Count of IOs']

# Grouping data by Scibids Activity and computing the mean for adjusted CPC and CPM
zetaglobal_cpc_cpm_per_io_summary = zetaglobal_rows.groupby('Performance Measures Billing Scibids Activity')[['CPC_per_IO', 'CPM_per_IO']].mean()

zetaglobal_cpc_cpm_per_io_summary

# Filter data for Scibids active campaigns
scibids_active_data = zetaglobal_rows[zetaglobal_rows['Performance Measures Billing Scibids Activity'] == 'Scibids Active']

# Group by month and calculate average CPC and CPM
monthly_avg = scibids_active_data.groupby('Performance Measures Day Tz Month')[['CPC', 'CPM']].mean()
monthly_avg.reset_index(inplace=True)

# Plotting
plt.figure(figsize=(14, 6))

# Plotting CPC
plt.subplot(1, 2, 1)
plt.plot(monthly_avg['Performance Measures Day Tz Month'], monthly_avg['CPC'], marker='o', color='b', label='CPC')
plt.xticks(rotation=45)
plt.title("Average CPC Over Time")
plt.xlabel("Month")
plt.ylabel("Average CPC")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Plotting CPM
plt.subplot(1, 2, 2)
plt.plot(monthly_avg['Performance Measures Day Tz Month'], monthly_avg['CPM'], marker='o', color='r', label='CPM')
plt.xticks(rotation=45)
plt.title("Average CPM Over Time")
plt.xlabel("Month")
plt.ylabel("Average CPM")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()

# Grouping by month, client ID, advertiser name, and insertion orders
grouped_cpc = scibids_active_data.groupby(['Performance Measures Day Tz Month', 'Accessible IDs Client ID',
                                           'Accessible IDs Advertiser Name', 'Insertion Orders Distinct Count of IOs'])['CPC'].mean().reset_index()

# Determining the 90th percentile as threshold
threshold = grouped_cpc['CPC'].quantile(0.90)

# Filtering campaigns with CPC greater than the threshold
high_cpc_campaigns = grouped_cpc[grouped_cpc['CPC'] > threshold]

high_cpc_campaigns

"""# Seasonal trends

## Campaigns trends
"""

# Grouping by month to get the total number of campaigns over time
campaigns_trend = scibids_active.groupby('Performance Measures Day Tz Month')['Insertion Orders Distinct Count of IOs'].sum()

# Plotting
plt.figure(figsize=(15, 7))
campaigns_trend.plot(kind='line', marker='o', color='mediumvioletred')
plt.title('Trend of Number of Campaigns Over Time')
plt.xlabel('Time (Month)')
plt.ylabel('Number of Campaigns')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Grouping by month, KPI, DSP, and region to get the sum of Insertion Orders and Revenue
combined_trends = scibids_active.groupby(['Performance Measures Day Tz Month',
                                         'unified_KPI',
                                         'Accessible IDs Dsp',
                                         'Clients Characteristics Scibids Region'])[['Insertion Orders Distinct Count of IOs',
                                                                                    'Performance Measures Revenue USD']].sum().reset_index()

# Normalizing the Insertion Orders and Revenue columns by month
combined_trends['Normalized IOs'] = combined_trends.groupby('Performance Measures Day Tz Month')['Insertion Orders Distinct Count of IOs'].apply(lambda x: x / x.sum() * 100)
combined_trends['Normalized Revenue'] = combined_trends.groupby('Performance Measures Day Tz Month')['Performance Measures Revenue USD'].apply(lambda x: x / x.sum() * 100)

# Selecting a subset of columns for clarity
combined_trends = combined_trends[['Performance Measures Day Tz Month', 'unified_KPI', 'Accessible IDs Dsp', 'Clients Characteristics Scibids Region', 'Normalized IOs', 'Normalized Revenue']]

combined_trends.head(10)  # Displaying the first 10 rows for a quick overview

"""## DSP trends on insertion orders and revenue"""

# Exploring trends in specific DSPs over time.

# Grouping by month and DSP for campaign counts and revenue
campaigns_per_month_dsp = scibids_active.groupby(['Performance Measures Day Tz Month', 'Accessible IDs Dsp'])['Insertion Orders Distinct Count of IOs'].sum()
revenue_per_month_dsp = scibids_active.groupby(['Performance Measures Day Tz Month', 'Accessible IDs Dsp'])['Performance Measures Revenue USD'].sum()

# Unstacking for visualization
campaigns_per_month_dsp_unstacked = campaigns_per_month_dsp.unstack('Accessible IDs Dsp').fillna(0)
revenue_per_month_dsp_unstacked = revenue_per_month_dsp.unstack('Accessible IDs Dsp').fillna(0)

campaigns_per_month_dsp_unstacked, revenue_per_month_dsp_unstacked

# Grouping by month, DSP, and region to get the sum of Insertion Orders and Revenue
dsp_trends = scibids_active.groupby(['Performance Measures Day Tz Month',
                                     'Accessible IDs Dsp',
                                     'Clients Characteristics Scibids Region'])[['Insertion Orders Distinct Count of IOs',
                                                                                'Performance Measures Revenue USD']].sum().reset_index()

# Normalizing the Insertion Orders and Revenue columns by month
dsp_trends['Normalized IOs'] = dsp_trends.groupby('Performance Measures Day Tz Month')['Insertion Orders Distinct Count of IOs'].apply(lambda x: x / x.sum() * 100)
dsp_trends['Normalized Revenue'] = dsp_trends.groupby('Performance Measures Day Tz Month')['Performance Measures Revenue USD'].apply(lambda x: x / x.sum() * 100)

# Selecting a subset of columns for clarity
dsp_trends = dsp_trends[['Performance Measures Day Tz Month', 'Accessible IDs Dsp', 'Clients Characteristics Scibids Region', 'Normalized IOs', 'Normalized Revenue']]

dsp_trends

# Setting up the figure for DSP trends based on normalized IOs
plt.figure(figsize=(18, 15))

# Creating a line plot for each region based on normalized IOs for DSPs
for idx, region in enumerate(regions, start=1):
    plt.subplot(3, 2, idx)
    region_data = dsp_trends[dsp_trends['Clients Characteristics Scibids Region'] == region]
    sns.lineplot(data=region_data, x='Performance Measures Day Tz Month', y='Normalized IOs', hue='Accessible IDs Dsp', ci=None)
    plt.title(f'DSP Trends in {region} (Normalized IOs)')
    plt.xticks(rotation=45)
    plt.ylabel('Normalized IOs (%)')
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Setting up the figure for DSP trends based on normalized revenue
regions = kpi_trends['Clients Characteristics Scibids Region'].unique()


plt.figure(figsize=(18, 15))

# Creating a line plot for each region based on normalized revenue for DSPs
for idx, region in enumerate(regions, start=1):
    plt.subplot(3, 2, idx)
    region_data = dsp_trends[dsp_trends['Clients Characteristics Scibids Region'] == region]
    sns.lineplot(data=region_data, x='Performance Measures Day Tz Month', y='Normalized Revenue', hue='Accessible IDs Dsp', ci=None)
    plt.title(f'DSP Trends in {region} (Normalized Revenue)')
    plt.xticks(rotation=45)
    plt.ylabel('Normalized Revenue (%)')
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Plotting the seasonal trends based on revenue for each DSP in each market


plt.figure(figsize=(18, 12))

for dsp in dsps:
    dsp_subset = dsp_data[dsp_data['Accessible IDs Dsp'] == dsp]
    plt.subplot(3, 2, dsps.tolist().index(dsp) + 1)
    sns.lineplot(data=dsp_subset, x='Performance Measures Day Tz Month', y='Normalized Revenue', hue='Clients Characteristics Scibids Region', marker='o')
    plt.title(f'Revenue Trend for {dsp}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Plotting the revenue trend for each DSP over time

plt.figure(figsize=(14, 7))
for dsp in revenue_per_month_dsp_unstacked.columns:
    # Convert the 'Period' index to string
    plt.plot(revenue_per_month_dsp_unstacked.index.astype(str), revenue_per_month_dsp_unstacked[dsp], label=dsp, marker='o')

plt.title('Revenue Trend for Each DSP over Time')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility
plt.legend()
plt.show()

# Plotting the trend of campaigns for each DSP over time

plt.figure(figsize=(14, 7))
for dsp in campaigns_per_month_dsp_unstacked.columns:
    # Convert the 'Period' index to string
    plt.plot(campaigns_per_month_dsp_unstacked.index.astype(str), campaigns_per_month_dsp_unstacked[dsp], label=dsp, marker='o')

plt.title('Trend of Campaigns for Each DSP over Time')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility
plt.legend()
plt.show()

"""## Monthly Revenue Evolution for Scibids Active Grouped by Client Typology"""

# Grouping the data by Month and Client Typology for "Scibids Active"
revenue_per_month_active_typology = scibids_active.groupby(['Performance Measures Day Tz Month', 'Clients Characteristics Typology'])['Performance Measures Revenue USD'].sum()

# Unstacking the data for visualization by typology
revenue_per_month_df_active_typology = revenue_per_month_active_typology.unstack('Clients Characteristics Typology').fillna(0)

plt.figure(figsize=(15, 7))

# Looping through each typology to create a line
for typology in revenue_per_month_df_active_typology.columns:
    plt.plot(revenue_per_month_df_active_typology.index, revenue_per_month_df_active_typology[typology], label=typology)

plt.title('Monthly Revenue for Scibids Active Grouped by Client Typology')
plt.xlabel('Month')
plt.ylabel('Sum of Revenue USD')
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

"""## Monthly Revenue Evolution for Scibids Active Grouped by Region"""

# Grouping the data by Month and Region for "Scibids Active"
revenue_per_month_active_region = scibids_active.groupby(['Performance Measures Day Tz Month', 'Clients Characteristics Scibids Region'])['Performance Measures Revenue USD'].sum()

# Unstacking the data for visualization by region
revenue_per_month_df_active_region = revenue_per_month_active_region.unstack('Clients Characteristics Scibids Region').fillna(0)

plt.figure(figsize=(15, 7))

# Looping through each region to create a line
for region in revenue_per_month_df_active_region.columns:
    plt.plot(revenue_per_month_df_active_region.index, revenue_per_month_df_active_region[region], label=region)

plt.title('Monthly Revenue for Scibids Active Grouped by Region')
plt.xlabel('Month')
plt.ylabel('Sum of Revenue USD')
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

revenue_per_month_region = scibids_active.groupby(['Performance Measures Day Tz Month', 'Clients Characteristics Scibids Region', 'Clients Characteristics Typology', 'Insertion Orders Distinct Count of IOs'])['Performance Measures Revenue USD'].sum()
revenue_per_month_region



"""## KPI trends per insertion orders and revenue"""

# Exploring trends in specific KPIs over time.

# Grouping by month and KPI for campaign counts and revenue
campaigns_per_month_kpi = scibids_active.groupby(['Performance Measures Day Tz Month', 'Insertion Orders Kpi to Optimize'])['Insertion Orders Distinct Count of IOs'].sum()
revenue_per_month_kpi = scibids_active.groupby(['Performance Measures Day Tz Month', 'Insertion Orders Kpi to Optimize'])['Performance Measures Revenue USD'].sum()

# Unstacking for visualization
campaigns_per_month_kpi_unstacked = campaigns_per_month_kpi.unstack('Insertion Orders Kpi to Optimize').fillna(0)
revenue_per_month_kpi_unstacked = revenue_per_month_kpi.unstack('Insertion Orders Kpi to Optimize').fillna(0)

campaigns_per_month_kpi_unstacked, revenue_per_month_kpi_unstacked

# Grouping by month, KPI, and region to get the sum of Insertion Orders and Revenue
kpi_trends = scibids_active.groupby(['Performance Measures Day Tz Month',
                                     'unified_KPI',
                                     'Clients Characteristics Scibids Region'])[['Insertion Orders Distinct Count of IOs',
                                                                                'Performance Measures Revenue USD']].sum().reset_index()

# Normalizing the Insertion Orders and Revenue columns by month
kpi_trends['Normalized IOs'] = kpi_trends.groupby('Performance Measures Day Tz Month')['Insertion Orders Distinct Count of IOs'].apply(lambda x: x / x.sum() * 100)
kpi_trends['Normalized Revenue'] = kpi_trends.groupby('Performance Measures Day Tz Month')['Performance Measures Revenue USD'].apply(lambda x: x / x.sum() * 100)

# Selecting a subset of columns for clarity
kpi_trends = kpi_trends[['Performance Measures Day Tz Month', 'unified_KPI', 'Clients Characteristics Scibids Region', 'Normalized IOs', 'Normalized Revenue']]

kpi_trends

# List of unique regions in the dataset
regions = kpi_trends['Clients Characteristics Scibids Region'].unique()

# Setting up the figure
plt.figure(figsize=(18, 15))

# Creating a line plot for each region
for idx, region in enumerate(regions, start=1):
    plt.subplot(3, 2, idx)
    region_data = kpi_trends[kpi_trends['Clients Characteristics Scibids Region'] == region]
    sns.lineplot(data=region_data, x='Performance Measures Day Tz Month', y='Normalized IOs', hue='unified_KPI', ci=None)
    plt.title(f'KPI Trends in {region} (Normalized IOs)')
    plt.xticks(rotation=45)
    plt.ylabel('Normalized IOs (%)')
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Setting up the figure for normalized revenue plots
plt.figure(figsize=(18, 15))

# Creating a line plot for each region based on normalized revenue
for idx, region in enumerate(regions, start=1):
    plt.subplot(3, 2, idx)
    region_data = kpi_trends[kpi_trends['Clients Characteristics Scibids Region'] == region]
    sns.lineplot(data=region_data, x='Performance Measures Day Tz Month', y='Normalized Revenue', hue='unified_KPI', ci=None)
    plt.title(f'KPI Trends in {region} (Normalized Revenue)')
    plt.xticks(rotation=45)
    plt.ylabel('Normalized Revenue (%)')
    plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Plotting the trend of campaigns for each KPI over time

plt.figure(figsize=(14, 7))
for kpi in campaigns_per_month_kpi_unstacked.columns:
    # Convert the 'Period' index to string
    plt.plot(campaigns_per_month_kpi_unstacked.index.astype(str), campaigns_per_month_kpi_unstacked[kpi], label=kpi, marker='o')

plt.title('Trend of Campaigns for Each KPI over Time')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.legend()
plt.show()

# Exploring trends in KPIs over time

# Grouping by month and KPI to get the total number of campaigns and revenue for each KPI over time
kpi_trends = df[df['Performance Measures Billing Scibids Activity'] == 'Scibids Active'].groupby(['Performance Measures Day Tz Month', 'unified_KPI'])[['Insertion Orders Distinct Count of IOs', 'Performance Measures Revenue USD']].sum()

# Unstacking the KPI values for visualization
kpi_trends_unstacked = kpi_trends['Insertion Orders Distinct Count of IOs'].unstack().fillna(0)

# Plotting the trends for the top 5 KPIs based on the number of campaigns
plt.figure(figsize=(15, 7))
top_kpis = kpi_trends_unstacked.sum().nlargest(5).index
kpi_trends_unstacked[top_kpis].plot(ax=plt.gca())
plt.title('Trends in Top 5 KPIs Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Campaigns')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.legend(title='KPIs')
plt.show()

# Plotting the revenue trend for each KPI over time

plt.figure(figsize=(14, 7))
for kpi in revenue_per_month_kpi_unstacked.columns:
    # Convert the 'Period' index to string
    plt.plot(revenue_per_month_kpi_unstacked.index.astype(str), revenue_per_month_kpi_unstacked[kpi], label=kpi, marker='o')

plt.title('Revenue Trend for Each KPI over Time')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.legend()
plt.show()

"""# Performance measures

## Weighted revenue, CPM, CPC by typology Scibids active
"""

# Calculate average performance measures by typology, taking into account distinct order ids
typology_segmentation_weighted = df[df['Performance Measures Billing Scibids Activity'] == 'Scibids Active'].groupby('Clients Characteristics Typology').apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
)

# Sort by 'Weighted Revenue' for better visualization
typology_segmentation_weighted = typology_segmentation_weighted.sort_values(by='Weighted Revenue', ascending=False)

typology_segmentation_weighted



"""## Weighted revenue, CPM, CPC by typology groupoed by region"""

# Calculate average performance measures by region and typology, taking into account distinct order ids
region_typology_segmentation_weighted = scibids_active.groupby(['Clients Characteristics Scibids Region', 'Clients Characteristics Typology']).apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
)

# Sort by 'Weighted Revenue' for better visualization
region_typology_segmentation_weighted = region_typology_segmentation_weighted.sort_values(by='Weighted Revenue', ascending=False)

region_typology_segmentation_weighted

"""## Client and advertiser details"""

# Grouping by typology, region, DSP, sub DSP, company name, client name, and advertiser name
# and calculating the weighted measures
summary_counts = scibids_active.groupby(['Clients Characteristics Typology',
                                         'Clients Characteristics Scibids Region',
                                         'Accessible IDs Dsp',
                                         'Accessible IDs Sub Dsp',
                                         'Clients Characteristics Company Name',
                                         'Accessible IDs Client Name',
                                         'Accessible IDs Advertiser Name']).apply(
    lambda df: pd.Series({
        'Count': df.shape[0],
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
).reset_index()

summary_counts

"""## Weighted revenue, CPM, CPC by typology without Scibids"""

# Calculate average performance measures by typology, taking into account distinct order ids
typology_segmentation_weighted_without = df[df['Performance Measures Billing Scibids Activity'] == 'Without Scibids'].groupby('Clients Characteristics Typology').apply(
    lambda df: pd.Series({
        'Weighted Revenue 2': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC 2': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM 2': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
)


typology_segmentation_weighted_without

"""## Weighted revenue, CPM, CPC by typology grouped by region Without Scibids"""



# Without

# Calculate average performance measures by region and typology, taking into account distinct order ids
region_typology_without_weighted = df[df['Performance Measures Billing Scibids Activity'] == 'Without Scibids'].groupby(['Clients Characteristics Scibids Region', 'Clients Characteristics Typology']).apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
)

# Sort by 'Weighted Revenue' for better visualization
region_typology_without_weighted = region_typology_without_weighted.sort_values(by='Weighted Revenue', ascending=False)

region_typology_without_weighted

"""## Comparison with and without Scibids by typology and region"""

# Concatenate the two dataframes with a multi-level column index
performance_combined = pd.concat([region_typology_without_weighted, region_typology_segmentation_weighted], axis=1, keys=['With Scibids', 'Without Scibids'])

performance_combined

performance_combined.to_excel('performance_combined.xlsx')
files.download('performance_combined.xlsx')

"""## Exploring the unusually high values by checking distribution"""

# unusually high values

# List of columns to inspect
scibids_active

columns_to_inspect = [
     'Performance Measures Revenue USD',
    'CPC',
    'CPM',
    'Insertion Orders Distinct Count of IOs'

]

# Plotting the distributions for each column grouped by typology
fig, axes = plt.subplots(nrows=len(columns_to_inspect), ncols=1, figsize=(15, 20))

for i, col in enumerate(columns_to_inspect):
    scibids_active.boxplot(column=col, by='Clients Characteristics Typology', ax=axes[i], vert=False)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Client Typology')

plt.tight_layout()
plt.suptitle('Distribution of Metrics by Client Typology', y=1.02)
plt.show()

"""## Detect outliers Scibids active"""

# Function to detect outliers based on the IQR method

# With Scibids

def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Detecting outliers for the key metrics
outliers_cpc_with = detect_outliers(scibids_active, 'CPC')
outliers_cpm_with = detect_outliers(scibids_active, 'CPM')
outliers_revenue_with = detect_outliers(scibids_active, 'Performance Measures Revenue USD')

outliers_cpc_with, outliers_cpm_with, outliers_revenue_with

"""## Detect outliers without Scibids"""

# Function to detect outliers based on the IQR method

# Without Scibids

def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Detecting outliers for the key metrics
outliers_cpc_without = detect_outliers(df[df['Performance Measures Billing Scibids Activity'] == 'Without Scibids'], 'CPC')
outliers_cpm_without = detect_outliers(df[df['Performance Measures Billing Scibids Activity'] == 'Without Scibids'], 'CPM')
outliers_revenue_without = detect_outliers(df[df['Performance Measures Billing Scibids Activity'] == 'Without Scibids'], 'Performance Measures Revenue USD')

outliers_cpc_without, outliers_cpm_without, outliers_revenue_without

outliers_cpc_without_count = outliers_cpc_without.shape[0]
outliers_cpm_without_count = outliers_cpm_without.shape[0]
outliers_revenue_without_count = outliers_revenue_without.shape[0]

outliers_cpc_without_count, outliers_cpm_without_count, outliers_revenue_without_count



outliers_cpc_with_count = outliers_cpc_with.shape[0]
outliers_cpm_with_count = outliers_cpm_with.shape[0]
outliers_revenue_with_count = outliers_revenue_with.shape[0]

outliers_cpc_with_count, outliers_cpm_with_count, outliers_revenue_with_count

# Plotting the distribution of outliers for each metric
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 18))

# Plotting outliers for CPC
outliers_cpc_with['CPC'].plot(kind='hist', bins=50, ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Distribution of Outliers for CPC')
axes[0].set_xlabel('CPC')
axes[0].set_ylabel('Frequency')

# Plotting outliers for CPM
outliers_cpm_with['CPM'].plot(kind='hist', bins=50, ax=axes[1], color='salmon', edgecolor='black')
axes[1].set_title('Distribution of Outliers for CPM')
axes[1].set_xlabel('CPM')
axes[1].set_ylabel('Frequency')

# Plotting outliers for Revenue
outliers_revenue_with['Performance Measures Revenue USD'].plot(kind='hist', bins=50, ax=axes[2], color='lightgreen', edgecolor='black')
axes[2].set_title('Distribution of Outliers for Performance Measures Revenue USD')
axes[2].set_xlabel('Performance Measures Revenue USD')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

"""## Outliers by typology and metric"""

# Getting counts of outliers by client typology for each metric
outliers_cpc_typology = outliers_cpc_with['Clients Characteristics Typology'].value_counts()
outliers_cpm_typology = outliers_cpm_with['Clients Characteristics Typology'].value_counts()
outliers_revenue_typology = outliers_revenue_with['Clients Characteristics Typology'].value_counts()

outliers_cpc_typology, outliers_cpm_typology, outliers_revenue_typology

# Combining the counts into a single DataFrame for easier visualization
outliers_by_typology = pd.DataFrame({
    'CPC Outliers': outliers_cpc_typology,
    'CPM Outliers': outliers_cpm_typology,
    'Revenue Outliers': outliers_revenue_typology
}).fillna(0).astype(int)

# Sorting by total outliers across all metrics
outliers_by_typology['Total Outliers'] = outliers_by_typology.sum(axis=1)
outliers_by_typology = outliers_by_typology.sort_values(by='Total Outliers', ascending=False)

outliers_by_typology

# Count outliers per DSP
dsp_outliers_counts = {
    'CPC Outliers': outliers_cpc_with['Accessible IDs Dsp'].value_counts(),
    'CPM Outliers': outliers_cpm_with['Accessible IDs Dsp'].value_counts(),
    'Revenue Outliers': outliers_revenue_with['Accessible IDs Dsp'].value_counts()
}

# Count outliers per Region
region_outliers_counts = {
    'CPC Outliers': outliers_cpc_with['Clients Characteristics Scibids Region'].value_counts(),
    'CPM Outliers': outliers_cpm_with['Clients Characteristics Scibids Region'].value_counts(),
    'Revenue Outliers': outliers_revenue_with['Clients Characteristics Scibids Region'].value_counts()
}

# Combine counts into DataFrames for easier visualization
outliers_by_dsp = pd.DataFrame(dsp_outliers_counts).fillna(0).astype(int)
outliers_by_region = pd.DataFrame(region_outliers_counts).fillna(0).astype(int)

outliers_by_dsp

outliers_by_region

# Combine the outliers dataframes
combined_outliers = pd.concat([outliers_cpc_with, outliers_cpm_with, outliers_revenue_with]).drop_duplicates()

# Grouping by typology, region, DSP, sub DSP, company name, client name, and advertiser name
# and calculating the weighted measures for the outliers
summary_counts_outliers = combined_outliers.groupby(['Clients Characteristics Typology',
                                                     'Clients Characteristics Scibids Region',
                                                     'Accessible IDs Dsp',
                                                     'Accessible IDs Sub Dsp',
                                                     'Clients Characteristics Company Name',
                                                     'Accessible IDs Client Name',
                                                     'Accessible IDs Advertiser Name']).apply(
    lambda df: pd.Series({
        'Count': df.shape[0],
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
).reset_index()

summary_counts_outliers

from google.colab import files
summary_counts_outliers.to_csv('summary_counts_outliers.csv')
files.download('summary_counts_outliers.csv')

# Isolating the top 1% outliers for each metric to inspect them in more detail
top_1_percent_cpc = outliers_cpc_with.nlargest(int(0.01 * len(outliers_cpc_with)), 'CPC')
top_1_percent_cpm = outliers_cpm_with.nlargest(int(0.01 * len(outliers_cpm_with)), 'CPM')
top_1_percent_revenue = outliers_revenue_with.nlargest(int(0.01 * len(outliers_revenue_with)), 'Performance Measures Revenue USD')

# Checking the basic statistics of the top 1% outliers
top_outliers_stats = {
    'CPC Top 1%': top_1_percent_cpc['CPC'].describe(),
    'CPM Top 1%': top_1_percent_cpm['CPM'].describe(),
    'Revenue Top 1%': top_1_percent_revenue['Performance Measures Revenue USD'].describe()
}

pd.DataFrame(top_outliers_stats)

"""## Outliers by region and DSP"""

# Grouping the top outliers by Region and DSP to see the counts
region_outliers_counts = {
    'CPC Top 1%': top_1_percent_cpc['Clients Characteristics Scibids Region'].value_counts(),
    'CPM Top 1%': top_1_percent_cpm['Clients Characteristics Scibids Region'].value_counts(),
    'Revenue Top 1%': top_1_percent_revenue['Clients Characteristics Scibids Region'].value_counts()
}

dsp_outliers_counts = {
    'CPC Top 1%': top_1_percent_cpc['Accessible IDs Dsp'].value_counts(),
    'CPM Top 1%': top_1_percent_cpm['Accessible IDs Dsp'].value_counts(),
    'Revenue Top 1%': top_1_percent_revenue['Accessible IDs Dsp'].value_counts()
}

# Combining the counts into DataFrames for easier visualization
outliers_by_region = pd.DataFrame(region_outliers_counts).fillna(0).astype(int)
outliers_by_dsp = pd.DataFrame(dsp_outliers_counts).fillna(0).astype(int)

outliers_by_region

outliers_by_dsp

"""## Top clients"""

# Group by Company and sum the Revenue
top_companies = df.groupby('Clients Characteristics Company Name')['Performance Measures Revenue USD'].sum().sort_values(ascending=False).head(10)

# Group by Client and sum the Revenue
top_clients = df.groupby('Accessible IDs Client Name')['Performance Measures Revenue USD'].sum().sort_values(ascending=False).head(10)

# Group by Advertiser and sum the Revenue
top_advertisers = df.groupby('Accessible IDs Advertiser Name')['Performance Measures Revenue USD'].sum().sort_values(ascending=False).head(10)

top_companies, top_clients, top_advertisers

# Calculate the weighted revenue, CPC, and CPM for each row
df['Weighted_Revenue'] = df['Performance Measures Revenue USD'] / df['Insertion Orders Distinct Count of IOs']
df['Weighted_CPC'] = df['CPC'] / df['Insertion Orders Distinct Count of IOs']
df['Weighted_CPM'] = df['CPM'] / df['Insertion Orders Distinct Count of IOs']

# Grouping by hierarchy: Region > Company > Client > Advertiser and summing the weighted metrics
grouped_weighted_values_by_region = df.groupby(
    ['Clients Characteristics Scibids Region', 'Clients Characteristics Company Name', 'Accessible IDs Client Name', 'Accessible IDs Advertiser Name', 'Clients Characteristics Typology', 'Insertion Orders Distinct Count of IOs']
).agg({
    'Weighted_Revenue': 'sum',
    'Weighted_CPC': 'sum',
    'Weighted_CPM': 'sum'
}).reset_index()

# Sorting to get the top entities based on Weighted Revenue for each region
grouped_weighted_values_by_region = grouped_weighted_values_by_region.sort_values(by=['Clients Characteristics Scibids Region', 'Weighted_Revenue', 'Clients Characteristics Typology', 'Insertion Orders Distinct Count of IOs'], ascending=[False, False, False, False])

# Displaying the top 10 for each region (if you have fewer regions, you might want to adjust this)
top_entities_by_region = grouped_weighted_values_by_region.groupby('Clients Characteristics Scibids Region').head()
top_entities_by_region

"""## Weighted measures outliners by typology with and Without Scibids"""

# Concatenate outlier dataframes
all_outliers_with = pd.concat([outliers_cpc_with, outliers_cpm_with, outliers_revenue_with])

# Group by 'Clients Characteristics Typology' and calculate the weighted metrics
outliers_weighted_with = all_outliers_with.groupby('Clients Characteristics Typology').apply(
    lambda df: pd.Series({
        'Weighted Revenue 4': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC 4': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM 4': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
).reset_index()

outliers_weighted_with

# Concatenate outlier dataframes
all_outliers_without = pd.concat([outliers_cpc_without, outliers_cpm_without, outliers_revenue_without])

# Group by 'Clients Characteristics Typology' and calculate the weighted metrics
outliers_weighted_without = all_outliers_without.groupby('Clients Characteristics Typology').apply(
    lambda df: pd.Series({
        'Weighted Revenue 3': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC 3': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM 3': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
).reset_index()

outliers_weighted_without

# Concatenate the two dataframes with a multi-level column index
concatenated_weighted_outliners = pd.concat([outliers_weighted_with, outliers_weighted_without], axis=1, keys=['With Scibids', 'Without Scibids'])

concatenated_weighted_outliners

"""## Dsitribution of low and high outliers per region

To identify outliers based on regional benchmarks instead of the overall dataset, we will calculate the quantiles for revenue, CPC, and CPM within each specific region. This will provide a more accurate and regionally relevant understanding of what constitutes an outlier.
 This ensures that the identification of outliers is based on the specific characteristics of each region, which is more meaningful and insightful for regional analysis.
"""

# Extract the unique regions from the dataset
regions = summary_counts_outliers['Clients Characteristics Scibids Region'].unique()

# Initialize dictionaries to store high and low outliers for each region
high_outliers_per_region = {}
low_outliers_per_region = {}

# Loop through each region to find the high and low outliers
for region in regions:
    region_data = summary_counts_outliers[summary_counts_outliers['Clients Characteristics Scibids Region'] == region]

    # Calculate the quantiles for the region
    high_quantile = region_data['Weighted Revenue'].quantile(0.75)
    low_quantile = region_data['Weighted Revenue'].quantile(0.25)

    # Find the top 5 high outliers for the region
    high_outliers = region_data[region_data['Weighted Revenue'] > high_quantile].sort_values(by='Weighted Revenue', ascending=False).head(5)
    high_outliers_per_region[region] = high_outliers

    # Find the top 5 low outliers for the region
    low_outliers = region_data[region_data['Weighted Revenue'] < low_quantile].sort_values(by='Weighted Revenue').head(5)
    low_outliers_per_region[region] = low_outliers

# Display the top 5 high and low outliers for each region
high_outliers_per_region, low_outliers_per_region

"""## % Difference with and Without Scibids outliners"""



# Calculate percentage differences for each metric using the adjusted column names
dff_adjusted['Revenue % Difference'] = ((dff_adjusted['Weighted Revenue With Scibids'] - dff_adjusted['Weighted Revenue Without Scibids']) / dff_adjusted['Weighted Revenue Without Scibids']) * 100
dff_adjusted['CPC % Difference'] = ((dff_adjusted['Weighted CPC With Scibids'] - dff_adjusted['Weighted CPC Without Scibids']) / dff_adjusted['Weighted CPC Without Scibids']) * 100
dff_adjusted['CPM % Difference'] = ((dff_adjusted['Weighted CPM With Scibids'] - dff_adjusted['Weighted CPM Without Scibids']) / dff_adjusted['Weighted CPM Without Scibids']) * 100

# Return the dataframe with percentage differences
dff_adjusted[['Clients Characteristics Typology', 'Revenue % Difference', 'CPC % Difference', 'CPM % Difference']]



## Details on top outliners based on revenue, CPC, and CPM for a global overview
"""

# Extracting relevant columns for the top outliers
columns_of_interest = [
    'Clients Characteristics Company Name',
    'Accessible IDs Client Name',
    'Accessible IDs Advertiser Name',
    'Clients Characteristics Scibids Region',
    'Clients Characteristics Typology',
    'Accessible IDs Dsp',
    'Accessible IDs Sub Dsp',
    'unified_KPI',
    'Insertion Orders Distinct Count of IOs'
]

# Extracting the details for the top 1% outliers in each metric
cpc_outliers_details = top_1_percent_cpc[columns_of_interest]


cpc_outliers_details

# Extracting the details for the top 1% outliers in each metric
cpc_outliers_details = top_1_percent_cpc[columns_of_interest]
cpm_outliers_details = top_1_percent_cpm[columns_of_interest]



cpm_outliers_details

# Extracting the details for the top 1% outliers in each metric

revenue_outliers_details = top_1_percent_revenue[columns_of_interest]


revenue_outliers_details

"""## Top outliner count by region and typology"""

# Concatenating the top 1% outliers for all metrics into a single dataframe
all_top_outliers = pd.concat([top_1_percent_cpc, top_1_percent_cpm, top_1_percent_revenue])

# Grouping by typology and region
grouped_outliers = all_top_outliers.groupby(['Clients Characteristics Typology', 'Clients Characteristics Scibids Region']).size().reset_index(name='Outlier Count')

# Sorting for better visualization
grouped_outliers = grouped_outliers.sort_values(by='Outlier Count', ascending=False)

grouped_outliers

"""## Weighted measures without outliners Scibids active




"""

# Removing the top 1% outliers from the main dataframe
cleaned_data_without_outliers = scibids_active[~scibids_active.index.isin(all_top_outliers.index)]

# Recalculate the weighted measures
weighted_measures_without_outliners = cleaned_data_without_outliers.groupby('Clients Characteristics Typology').apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
).reset_index()

# Sorting by 'Weighted Revenue' for better visualization
weighted_measures_without_outliners = weighted_measures_without_outliners.sort_values(by='Weighted Revenue', ascending=False)

weighted_measures_without_outliners

# Calculate average performance measures by region and typology, taking into account distinct order ids
region_weighted_measures_without_outliners = cleaned_data_without_outliers.groupby(['Clients Characteristics Scibids Region', 'Clients Characteristics Typology']).apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
)

# Sort by 'Weighted Revenue' for better visualization
region_weighted_measures_without_outliners =region_weighted_measures_without_outliners.sort_values(by='Weighted Revenue', ascending=False)
region_weighted_measures_without_outliners

"""## Weighted measures without outliners without Scibids

"""

# Without Scibids removing outliners
# Removing the top 1% outliers from the main dataframe
cleaned_data_without_outliers_without_scibids = df[df['Performance Measures Billing Scibids Activity'] == 'Without Scibids'][~df[df['Performance Measures Billing Scibids Activity'] == 'Without Scibids'].index.isin(all_top_outliers.index)]

# Recalculate the weighted measures
weighted_measures_without_outliners_without_scibids = cleaned_data_without_outliers_without_scibids.groupby('Clients Characteristics Typology').apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
).reset_index()

# Sorting by 'Weighted Revenue' for better visualization
weighted_measures_without_outliners_without_scibids = weighted_measures_without_outliners_without_scibids.sort_values(by='Weighted Revenue', ascending=False)

weighted_measures_without_outliners_without_scibids

# Calculate average performance measures by region and typology, taking into account distinct order ids
region_weighted_measures_without_outliners_without_scibids = cleaned_data_without_outliers_without_scibids.groupby(['Clients Characteristics Scibids Region', 'Clients Characteristics Typology']).apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
)

# Sort by 'Weighted Revenue' for better visualization
region_weighted_measures_without_outliners_without_scibids =region_weighted_measures_without_outliners_without_scibids.sort_values(by='Weighted Revenue', ascending=False)
region_weighted_measures_without_outliners_without_scibids

"""## Weighted measures without outliners compared on Scibids activity

"""

# Concatenate the two dataframes with a multi-level column index
concatenated_weighted_no_outliners = pd.concat([weighted_measures_without_outliners, weighted_measures_without_outliners_without_scibids], axis=1, keys=['With Scibids', 'Without Scibids'])

concatenated_weighted_no_outliners

# Concatenate the two dataframes with a multi-level column index
performance_combined_no_outliers = pd.concat([region_weighted_measures_without_outliners, region_weighted_measures_without_outliners_without_scibids], axis=1, keys=['With Scibids', 'Without Scibids'])

performance_combined_no_outliers



"""## % Difference without outliners based on Scibids activity"""

from pandas.core.api import DateOffset
# Calculate the differences between the values in the two tables




# Calculate percentage differences for each metric
df_without_outliers['Revenue % Difference'] = ((df_without_outliers['Weighted Revenue With Scibids'] - df_without_outliers['Weighted Revenue Without Scibids']) / df_without_outliers['Weighted Revenue Without Scibids']) * 100
df_without_outliers['CPC % Difference'] = ((df_without_outliers['Weighted CPC With Scibids'] - df_without_outliers['Weighted CPC Without Scibids']) / df_without_outliers['Weighted CPC Without Scibids']) * 100
df_without_outliers['CPM % Difference'] = ((df_without_outliers['Weighted CPM With Scibids'] - df_without_outliers['Weighted CPM Without Scibids']) / df_without_outliers['Weighted CPM Without Scibids']) * 100

# Return the dataframe with percentage differences
df_without_outliers[['Clients Characteristics Typology', 'Revenue % Difference', 'CPC % Difference', 'CPM % Difference']]

# Calculate percentage differences for each metric
df_with_outliers['Revenue % Difference'] = ((df_with_outliers['Weighted Revenue With Scibids'] - df_with_outliers['Weighted Revenue Without Scibids']) / df_with_outliers['Weighted Revenue Without Scibids']) * 100
df_with_outliers['CPC % Difference'] = ((df_with_outliers['Weighted CPC With Scibids'] - df_with_outliers['Weighted CPC Without Scibids']) / df_with_outliers['Weighted CPC Without Scibids']) * 100
df_with_outliers['CPM % Difference'] = ((df_with_outliers['Weighted CPM With Scibids'] - df_with_outliers['Weighted CPM Without Scibids']) / df_with_outliers['Weighted CPM Without Scibids']) * 100

# Return the dataframe with percentage differences
df_with_outliers[['Clients Characteristics Typology', 'Revenue % Difference', 'CPC % Difference', 'CPM % Difference']]

"""Revenue % Difference:
A positive value indicates that the expenditure (or cost) with Scibids is higher than without Scibids.

CPC % Difference:
A positive value indicates that the CPC (Cost Per Click) with Scibids is higher than without Scibids, which is unfavorable. A lower CPC is preferred as it means each click costs less.

CPM % Difference:
A positive value indicates that the CPM (Cost Per Mille/Thousand) with Scibids is higher than without Scibids. A lower CPM is preferred as it means that a thousand impressions cost less.

"""

#  Analyzing how the presence of outliers affects the metrics

df_with_outliers = pd.DataFrame(data_with_outliers)
df_without_outliers = pd.DataFrame(data_without_outliers)

# Comparing the two dataframes
comparison = df_with_outliers.set_index('Clients Characteristics Typology') - df_without_outliers.set_index('Clients Characteristics Typology')

comparison

# Extracting the relevant columns for correlation analysis from the correct dataset reference
correlation_data = scibids_active[['Clients Characteristics Typology', 'Clients Characteristics Scibids Region', 'Accessible IDs Dsp',
                            'CPC', 'CPM', 'Performance Measures Revenue USD']]

# Encoding categorical variables for correlation analysis
correlation_data_encoded = pd.get_dummies(correlation_data, columns=['Clients Characteristics Typology', 'Clients Characteristics Scibids Region', 'Accessible IDs Dsp'])

# Calculating correlation matrix
correlation_matrix = correlation_data_encoded.corr()

# Extracting correlations of metrics with other features
cpc_correlations = correlation_matrix['CPC'].sort_values(ascending=False)
cpm_correlations = correlation_matrix['CPM'].sort_values(ascending=False)
revenue_correlations = correlation_matrix['Performance Measures Revenue USD'].sort_values(ascending=False)

cpc_correlations.head(10), cpm_correlations.head(10), revenue_correlations.head(10)

# Filtering for Scibids Active rows using the correct value

# Encoding categorical variables for correlation analysis on scibids_active_corrected dataframe
correlation_data_active_corrected = scibids_active[['Clients Characteristics Typology', 'Clients Characteristics Scibids Region', 'Accessible IDs Dsp',
                            'Performance Measures Revenue USD', 'Performance Measures Clicks', 'Performance Measures Impressions']]
correlation_data_encoded_active_corrected = pd.get_dummies(correlation_data_active_corrected, columns=['Clients Characteristics Typology', 'Clients Characteristics Scibids Region', 'Accessible IDs Dsp'])

# Calculating correlation matrix for scibids_active_corrected
correlation_matrix_active_corrected = correlation_data_encoded_active_corrected.corr()

# Extracting correlations of metrics with other features for scibids_active_corrected
cpc_active_corrected_correlations = correlation_matrix_active_corrected['Performance Measures Clicks'].sort_values(ascending=False)
cpm_active_corrected_correlations = correlation_matrix_active_corrected['Performance Measures Impressions'].sort_values(ascending=False)
revenue_active_corrected_correlations = correlation_matrix_active_corrected['Performance Measures Revenue USD'].sort_values(ascending=False)

cpc_active_corrected_correlations.head(10), cpm_active_corrected_correlations.head(10), revenue_active_corrected_correlations.head(10)




# Calculate average performance measures by region, taking into account distinct order ids
region_segmentation_weighted = scibids_active.groupby('Clients Characteristics Scibids Region').apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
)

# Sort by 'Weighted Revenue' for better visualization
region_segmentation_weighted = region_segmentation_weighted.sort_values(by='Weighted Revenue', ascending=False)

region_segmentation_weighted

"""## Performance measures excluding outliners Scibids active"""

# Function to remove outliers based on the IQR method
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Remove outliers for the key metrics
scibids_active_cleaned = remove_outliers(scibids_active, 'Performance Measures Revenue USD')
scibids_active_cleaned = remove_outliers(scibids_active_cleaned, 'CPC')
scibids_active_cleaned = remove_outliers(scibids_active_cleaned, 'CPM')

# Calculate average performance measures by region and typology for the cleaned data
region_typology_segmentation_weighted_cleaned = scibids_active_cleaned.groupby(['Clients Characteristics Scibids Region', 'Clients Characteristics Typology']).apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
)

# Sort by 'Weighted Revenue' for better visualization
region_typology_segmentation_weighted_cleaned = region_typology_segmentation_weighted_cleaned.sort_values(by='Weighted Revenue', ascending=False)

region_typology_segmentation_weighted_cleaned

"""## Comparative analysis"""

summary_with_outliers = region_typology_segmentation_weighted.describe()
summary_with_outliers

summary_without_outliers = region_typology_segmentation_weighted_cleaned.describe()
summary_without_outliers

# Comparison of datasets to determine if outliers benefit more from Scibids

# Calculate the mean Revenue Savings, CPC Difference, and CPM Difference for both datasets
mean_values_with_outliers = region_typology_segmentation_weighted[['Revenue Savings (%)', 'CPC Difference', 'CPM Difference']].mean()
mean_values_without_outliers = region_typology_segmentation_weighted_cleaned[['Revenue Savings (%)', 'CPC Difference', 'CPM Difference']].mean()

comparison_outliers = pd.DataFrame({
    'With Outliers': mean_values_with_outliers,
    'Without Outliers': mean_values_without_outliers
})

comparison_outliers

# Compute regional and typology averages for both datasets
regional_summary_with = region_typology_segmentation_weighted.groupby('Clients Characteristics Scibids Region').mean()
regional_summary_without = region_typology_segmentation_weighted_cleaned.groupby('Clients Characteristics Scibids Region').mean()

# Now, let's merge these summaries for a direct comparison
comparison_df = pd.merge(regional_summary_with, regional_summary_without, on='Clients Characteristics Scibids Region', suffixes=('_with_outliers', '_without_outliers'))

# Viewing the comparison DataFrame
comparison_df

"""Performance with Outliers:


"""



"""## Performance by DSP"""

# Calculate average performance measures by DSP, taking into account distinct order ids
dsp_segmentation_weighted = scibids_active.groupby('Accessible IDs Dsp').apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
)

# Sort by 'Weighted Revenue' for better visualization
dsp_segmentation_weighted = dsp_segmentation_weighted.sort_values(by='Weighted Revenue', ascending=False)

dsp_segmentation_weighted

# Calculate average performance measures by Sub-DSP, taking into account distinct order ids
sub_dsp_segmentation_weighted = scibids_active.groupby('Accessible IDs Sub Dsp').apply(
    lambda df: pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })
)

# Sort by 'Weighted Revenue' for better visualization
sub_dsp_segmentation_weighted = sub_dsp_segmentation_weighted.sort_values(by='Weighted Revenue', ascending=False)

sub_dsp_segmentation_weighted



# Identifying the top 5 advertisers based on revenue
top_advertisers = scibids_active.groupby('Accessible IDs Client Name')['Performance Measures Revenue USD'].sum().nlargest(5)

# Extracting performance metrics for these top advertisers
top_advertiser_performance = scibids_active[df['Accessible IDs Client Name'].isin(top_advertisers.index)].groupby('Accessible IDs Client Name')[['Performance Measures Clicks', 'Performance Measures Impressions']].sum()

# Merging with the revenue data to have a comprehensive dataframe
top_advertiser_analysis = top_advertiser_performance.merge(top_advertisers, left_index=True, right_index=True)

top_advertiser_analysis



## Performance measures by typology and DSP compared by Scibids activity


# Weighted measures function
def weighted_measures(df):
    return pd.Series({
        'Weighted Revenue': df['Performance Measures Revenue USD'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPC': df['CPC'].sum() / df['Insertion Orders Distinct Count of IOs'].sum(),
        'Weighted CPM': df['CPM'].sum() / df['Insertion Orders Distinct Count of IOs'].sum()
    })

# Calculate for scibids_active_data
typology_dsp_segmentation_weighted_scibids = scibids_active.groupby(['Clients Characteristics Typology', 'Accessible IDs Dsp']).apply(weighted_measures)

# Calculate for without_scibids_data
typology_dsp_segmentation_weighted_without_scibids = without_scibids = df[df['Performance Measures Billing Scibids Activity'] == 'Without Scibids'].groupby(['Clients Characteristics Typology', 'Accessible IDs Dsp']).apply(weighted_measures)

# Pivot table to compare Scibids Active vs Without Scibids
pivot_typology_dsp = pd.concat([typology_dsp_segmentation_weighted_scibids.add_prefix('Scibids Active - '),
                         typology_dsp_segmentation_weighted_without_scibids.add_prefix('Without Scibids - ')], axis=1)

pivot_typology_dsp

"""Insights:

Areas of Opportunity: 
""" 

# Visualise
# Extracting data
scibids_revenue = pivot_typology_dsp['Scibids Active - Weighted Revenue']
without_scibids_revenue = pivot_typology_dsp['Without Scibids - Weighted Revenue']

scibids_cpc = pivot_typology_dsp['Scibids Active - Weighted CPC']
without_scibids_cpc = pivot_typology_dsp['Without Scibids - Weighted CPC']

scibids_cpm = pivot_typology_dsp['Scibids Active - Weighted CPM']
without_scibids_cpm = pivot_typology_dsp['Without Scibids - Weighted CPM']

# Setting up the plots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 18))

# Plotting Revenue
scibids_revenue.plot(kind='bar', ax=axes[0], position=0, color='blue', width=0.4)
without_scibids_revenue.plot(kind='bar', ax=axes[0], position=1, color='red', width=0.4)
axes[0].set_title('Weighted Revenue Comparison')
axes[0].set_ylabel('Weighted Revenue')
axes[0].legend(["Scibids Active", "Without Scibids"])

# Plotting CPC
scibids_cpc.plot(kind='bar', ax=axes[1], position=0, color='blue', width=0.4)
without_scibids_cpc.plot(kind='bar', ax=axes[1], position=1, color='red', width=0.4)
axes[1].set_title('Weighted CPC Comparison')
axes[1].set_ylabel('Weighted CPC')
axes[1].legend(["Scibids Active", "Without Scibids"])

# Plotting CPM
scibids_cpm.plot(kind='bar', ax=axes[2], position=0, color='blue', width=0.4)
without_scibids_cpm.plot(kind='bar', ax=axes[2], position=1, color='red', width=0.4)
axes[2].set_title('Weighted CPM Comparison')
axes[2].set_ylabel('Weighted CPM')
axes[2].legend(["Scibids Active", "Without Scibids"])

plt.tight_layout()
plt.show()



"""## Performance measures over time by Typology, Region and Scibids activity"""

# Extracting month and year from 'Performance Measures Day Tz Month' to create a 'Period' column
df['Period'] = df['Performance Measures Day Tz Month'].dt.strftime('%Y-%m')


# Calculate for scibids_active_data
typology_region_period_segmentation_weighted_scibids = scibids_active = df[df['Performance Measures Billing Scibids Activity'] == 'Scibids Active'].groupby(['Clients Characteristics Typology', 'Clients Characteristics Scibids Region', 'Period']).apply(weighted_measures)

# Calculate for without_scibids_data
typology_region_period_segmentation_weighted_without_scibids = without_scibids = df[df['Performance Measures Billing Scibids Activity'] == 'Without Scibids'].groupby(['Clients Characteristics Typology', 'Clients Characteristics Scibids Region', 'Period']).apply(weighted_measures)

# Pivot table to compare Scibids Active vs Without Scibids
pivot_typology_region_period = pd.concat([typology_region_period_segmentation_weighted_scibids.add_prefix('Scibids Active - '),
                         typology_region_period_segmentation_weighted_without_scibids.add_prefix('Without Scibids - ')], axis=1)

pivot_typology_region_period

"""## Further analysis on CTR, Impressions and clicks, not as relevant and CPC and CPM"""

# Grouping by 'Clients Characteristics Typology'
summary_table_typology = df.groupby(['Clients Characteristics Typology', 'Performance Measures Billing Scibids Activity'])[['Performance Measures Revenue USD', 'Performance Measures Clicks', 'Performance Measures Impressions', 'Insertion Orders Distinct Count of IOs']].sum()

# Calculate Normalized Revenue
summary_table_typology['Normalized Revenue (USD per Distinct Order ID)'] = summary_table_typology['Performance Measures Revenue USD'] / summary_table_typology['Insertion Orders Distinct Count of IOs']

# Drop the column used for normalization to keep the table clean
summary_table_typology.drop('Performance Measures Revenue USD', axis=1, inplace=True)
summary_table_typology.drop('Insertion Orders Distinct Count of IOs', axis=1, inplace=True)

# Rename the columns for a cleaner look
summary_table_typology.rename(columns={
    'Performance Measures Clicks': 'Total Clicks',
    'Performance Measures Impressions': 'Total Impressions'
}, inplace=True)

# Print the table
print(summary_table_typology)

# Create a summary table
summary_table = df.groupby(['Clients Characteristics Scibids Region', 'Performance Measures Billing Scibids Activity'])[['Performance Measures Revenue USD', 'Performance Measures Clicks', 'Performance Measures Impressions', 'Insertion Orders Distinct Count of IOs']].sum()

# Calculate Normalized Revenue
summary_table['Normalized Revenue (USD per Distinct Order ID)'] = summary_table['Performance Measures Revenue USD'] / summary_table['Insertion Orders Distinct Count of IOs']

# Drop the column used for normalization to keep the table clean
summary_table.drop('Performance Measures Revenue USD', axis=1, inplace=True)
summary_table.drop('Insertion Orders Distinct Count of IOs', axis=1, inplace=True)

# Rename the columns for a cleaner look
summary_table.rename(columns={
    'Performance Measures Clicks': 'Total Clicks',
    'Performance Measures Impressions': 'Total Impressions'
}, inplace=True)

# Print the table
print(summary_table)

# Let's try with CTR calculations (Click Through Rate (CTR) for ads, which offers a percentage of users who click on an ad after viewing it)

# Calculate CTR for both Scibids Active and Scibids Inactive
df['CTR'] = (df['Performance Measures Clicks'] / df['Performance Measures Impressions']) * 100

# Segmenting CTR by Scibids Activity, Regions, DSPs, and Client Typologies
ctr_scibids_active = df[df['Performance Measures Billing Scibids Activity'] == 'Scibids Active']
ctr_scibids_inactive = df[df['Performance Measures Billing Scibids Activity'] == 'Scibids Inactive']

# Grouping to get average CTR
ctr_by_region_active = ctr_scibids_active.groupby('Clients Characteristics Scibids Region')['CTR'].mean()
ctr_by_dsp_active = ctr_scibids_active.groupby('Accessible IDs Dsp')['CTR'].mean()
ctr_by_typology_active = ctr_scibids_active.groupby('Clients Characteristics Typology')['CTR'].mean()

ctr_by_region_inactive = ctr_scibids_inactive.groupby('Clients Characteristics Scibids Region')['CTR'].mean()
ctr_by_dsp_inactive = ctr_scibids_inactive.groupby('Accessible IDs Dsp')['CTR'].mean()
ctr_by_typology_inactive = ctr_scibids_inactive.groupby('Clients Characteristics Typology')['CTR'].mean()

ctr_by_region_active, ctr_by_dsp_active, ctr_by_typology_active, ctr_by_region_inactive, ctr_by_dsp_inactive, ctr_by_typology_inactive

# Visualization of the CTR segmented data

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 18))

# Plotting CTR by Region for Scibids Active
ctr_by_region_active.dropna().sort_values().plot(kind='barh', ax=axes[0], color='blue', alpha=0.7)
axes[0].set_title('Average CTR by Region (Scibids Active)')
axes[0].set_xlabel('CTR (%)')
axes[0].set_ylabel('Region')

# Plotting CTR by DSP for Scibids Active
ctr_by_dsp_active.dropna().sort_values().plot(kind='barh', ax=axes[1], color='green', alpha=0.7)
axes[1].set_title('Average CTR by DSP (Scibids Active)')
axes[1].set_xlabel('CTR (%)')
axes[1].set_ylabel('DSP')

# Plotting CTR by Client Typology for Scibids Active
ctr_by_typology_active.dropna().sort_values().plot(kind='barh', ax=axes[2], color='red', alpha=0.7)
axes[2].set_title('Average CTR by Client Typology (Scibids Active)')
axes[2].set_xlabel('CTR (%)')
axes[2].set_ylabel('Client Typology')

plt.tight_layout()
plt.show()

"""For campaigns with Scibids Active:

The highest average CTR for regions is observed in the...
Among the DSPs...
In terms of client typologies...

# Final dataset download
"""

from google.colab import files
df.to_csv('Tableau_nettoyeVF.csv')
files.download('Tableau_nettoyeVF.csv')
