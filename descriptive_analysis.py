from numpy.lib.arraysetops import unique
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sectors_separation import sector
import os

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#.................Result Folder making.......................
if not os.path.exists('descriptive_analysis_results'):
    os.makedirs('descriptive_analysis_results')

#...........Reading the data............................
df = pd.read_excel('Bases_Final_ADS_Jun2021.xlsx')

renda_mean, massa_mean = df['renda_r'].mean(), df['massa_r'].mean()
df['renda_r'].fillna(value=renda_mean, inplace=True)
df['massa_r'].fillna(value=massa_mean, inplace=True)

#.......... data subsetting used for finding correlation matrix of southeast region..........
df_correlation = df[['com_se','ind_se','res_se', 'renda_r','pop_ocup_br', 'massa_r', 'du','pmc_a_se',
        'temp_max_se', 'temp_min_se', 'pmc_r_se', 'pim_se']]


#..............Correlation Matrix.....................................
np.triu(np.ones_like(df_correlation.corr()))
plt.figure(figsize=(10, 6))
mask = np.triu(np.ones_like(df_correlation.corr(), dtype=np.bool))
ax = sns.heatmap(df_correlation.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
ax.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=16)
plt.xticks(rotation =15)
plt.show()
ax.figure.savefig('descriptive_analysis_results/Correlation_matrix' + '.png', format="PNG" )

#.............. Measuring central tendencyies of features (Just Southeast region)...................
central_tendency_se = df_correlation.describe()
central_tendency_se.to_csv('descriptive_analysis_results/central_tendency_southeast_region.csv')

#................................Grouping...............................................
df_sectors =sector(df)

df_sectors['years'] = df_sectors['data_tidy'].dt.year
df_sectors['months'] = df_sectors['data_tidy'].dt.month
years_tick = unique(df_sectors['years'])

months_label = ['Jan','Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#..................Monthly_Consumption...............................
with sns.axes_style("whitegrid"), sns.color_palette('Spectral', df_sectors['Sectors'].nunique()):
    plt.figure(figsize=(10,5))
    ax=sns.barplot(data=df_sectors, x= 'months',y= 'Energy_consumption(Gwh)', hue= 'Sectors' , ci= None)
    plt.xticks( fontsize = 12)
    plt.yticks( fontsize = 12)
    plt.title('Monthly energy consumption', fontsize = 15)
    ax.set_xticklabels(months_label)
    ax.set_xlabel('Months',fontsize=15)
    ax.set_ylabel('Energy_consumption(Gwh)',fontsize=15)
    ax.legend(loc='best')
plt.show()
ax.figure.savefig('descriptive_analysis_results/Monthly_Consumption' + '.png', format="PNG" )

#............................Yearly_Consumption...........................
with sns.axes_style("whitegrid"), sns.color_palette('Spectral', df_sectors['Sectors'].nunique()):
    plt.figure(figsize=(10,5))
    ax=sns.lineplot(data=df_sectors, x= 'years',y= 'Energy_consumption(Gwh)', size= 'Sectors' , ci= None)
    plt.xticks( fontsize = 12)
    plt.yticks( fontsize = 12)
    plt.xticks(years_tick, rotation =12)
    plt.title('Yearly energy consumption',fontsize = 15)
    ax.set_xlabel('Years',fontsize=15)
    ax.set_ylabel('Energy_consumption(Gwh)',fontsize=15)    
plt.show()
ax.figure.savefig('descriptive_analysis_results/Yearly_Consumption' + '.png', format="PNG" )

#......................Sector-wise_Consumption......................................
with sns.axes_style("whitegrid"), sns.color_palette('Spectral', df_sectors['Sectors'].nunique()):
    plt.figure(figsize=(10,5))
    ax=sns.boxplot(data=df_sectors, x= 'Energy_consumption(Gwh)',y= 'Regions',hue= 'Sectors')
    plt.xticks( fontsize = 12)
    plt.yticks( fontsize = 12)
    plt.title('Sector-wise energy consumption',fontsize = 15)
    ax.set_xlabel('Energy_consumption(Gwh)',fontsize=15)
    ax.set_ylabel('Regions',fontsize=15)
plt.show()
ax.figure.savefig('descriptive_analysis_results/Sectorwise_Consumption' + '.png', format="PNG" )

#..................Region-wise_Consumption...........................................
with sns.axes_style("whitegrid"), sns.color_palette('Spectral', df_sectors['Regions'].nunique()):
    plt.figure(figsize=(10,5))
    ax=sns.barplot(data=df_sectors, x= 'Sectors',y= 'Energy_consumption(Gwh)',hue= 'Regions', ci=None, estimator=np.mean)
    plt.xticks( fontsize = 12)
    plt.yticks( fontsize = 12)
    plt.title('Region-wise energy consumption',fontsize = 15)
    ax.set_xlabel('Sectors',fontsize=15)
    ax.set_ylabel('Energy_consumption(Gwh)',fontsize=15)
plt.show()
ax.figure.savefig('descriptive_analysis_results/Regionwise_Consumption' + '.png', format="PNG" )
