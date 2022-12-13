# Rohpate_IA626-Project
# New-York Crime Analysis upon Graduation Outcomes
# Reading CSV - Crime
Code:

import pandas as pd

import numpy as np

dt = pd.read_csv("bd2.csv")

dt

![image](https://user-images.githubusercontent.com/115191692/207200277-6dde3ff2-e17c-4dbd-809c-65e26f83f873.png)

# Selecting only Arrests of Age group: 18-24

dt1=dt.loc[dt['AGE_GROUP'].isin(['18-24'])]

df1=pd.DataFrame(dt1)

df1

![image](https://user-images.githubusercontent.com/115191692/207200948-c6565d28-9c0d-4717-9c46-e6f689e65e81.png)

# Dropping unwanted columns and Selecting Particular time period

df1.drop(df1.columns[[2,4,6,7,8,9,10]], axis=1, inplace=True)

df1.drop(df1.columns[[2,3]], axis=1, inplace=True)

df1 = df1[(df1['ARREST_DATE'] > '2011-12-31') & (df1['ARREST_DATE'] <= '2014-12-31')]

![image](https://user-images.githubusercontent.com/115191692/207202359-a5126763-26f5-4801-aacf-13819bcb1e89.png)


# Assigning New Columns for further analysis

df1['ARREST_YEAR']=df.ARREST_DATE.dt.strftime('%Y')

df1['Weekday'],df1['Day'],df1['Month'],= df1.ARREST_DATE.dt.strftime('%A'),df1.ARREST_DATE.dt.strftime('%d'),df1.ARREST_DATE.dt.strftime('%b')

df1.head()

![image](https://user-images.githubusercontent.com/115191692/207203691-a2c9cb6b-2404-4e43-ac63-0bc36f0fbf24.png)

# Visual Representation of data

## Highest crime area using basemap and folium

### Basemap
from mpl_toolkits.basemap import Basemap

m = Basemap(projection='mill',llcrnrlat=25,urcrnrlat=49.5\llcrnrlon=-140,urcrnrlon=-50,resolution='l')

plt.figure(figsize=(25,17))

m.drawcountries() #for drawing country borders

m.drawstates()    #for drawing states borders

m.drawcoastlines()

#m.fillcontinents(color='#04BAE3', lake_color='#FFFFFF') #giving color 

lat = 40.586466

lon = -73.816522

x,y = m(lon,lat)

m.plot(x, y, 'ro', markersize=20, alpha=.8) #alpha is making your marker transparent

m.bluemarble() #With this it's make your map like from satellite but if you give colors it will not work

m.drawmapboundary(color = '#FFFFFF')

plt.show()

![image](https://user-images.githubusercontent.com/115191692/207207448-61519b9b-851e-4497-a71d-4dafbc4d8f83.png)


### Folium

import folium

from folium.plugins import HeatMap

map_hooray = folium.Map(location=[40.586466,-73.989282],zoom_start = 12, min_zoom=12) 

heat_df = df[df['ARREST_YEAR']==2013]

heat_df = heat_df[heat_df['OFNS_DESC']=='DANGEROUS DRUGS'] 

heat_df = heat_df[['Latitude', 'Longitude']] 
                                       
folium.CircleMarker([40.586466,-73.989282],radius=50,popup='Homicide',color='red').add_to(map_hooray) 
    
    
heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]

HeatMap(heat_data, radius=10).add_to(map_hooray) #Adding map_hooray to HeatMap

map_hooray 

![image](https://user-images.githubusercontent.com/115191692/207207842-bd4319b2-3e1c-4804-90dd-becd2c810924.png)


## No. of Crimes per day and day of week

import matplotlib.pyplot as plt

day_order = ["Monday", "Tuesday", "Wednesday","Thursday","Friday","Saturday","Sunday"]

fig,(ax1,ax2) = plt.subplots(2,1, figsize=[15,10])

sns.countplot("Weekday", data=df,order=day_order,palette="RdBu_r",edgecolor="black", ax = ax1)

ax1.set_xlabel("Days of the week", alpha=0.75)

ax1.set_ylabel("Number of crimes", alpha=0.75)

sns.countplot("Day", data=df1,palette="RdBu_r",edgecolor="black", ax = ax2)

ax2.set_xlabel("Day", alpha=0.75)

ax2.set_ylabel("Number of crimes", alpha=0.75)

![image](https://user-images.githubusercontent.com/115191692/207204370-7826ee57-a8ce-44e7-9e0e-6be0c30b49a6.png)

## No. of Crimes commited and  its Crime Type

ax,fig = plt.subplots(figsize=(15,7))



sns.countplot(y = df["OFNS_DESC"],order=df["OFNS_DESC"].value_counts()[:10].index, palette="RdBu_r",edgecolor="black")

plt.ylabel("Crime Type", fontsize=22, alpha=.75)

plt.xlabel("Number of crimes committed", fontsize=22, alpha=.75)

plt.yticks(alpha=0.75,weight="bold")

plt.xticks(alpha=0.75)

![image](https://user-images.githubusercontent.com/115191692/207204804-234920a0-068e-4b69-8c6c-c7f31bdb43bc.png)

## No. of Crimes per month

plt.figure(figsize=(8,8))

sns.countplot(x=df1.Month)

plt.show() 

![image](https://user-images.githubusercontent.com/115191692/207205074-503736e1-a729-4830-9472-58156fb33fc5.png)

## Crime Vs Year Chart

year_count = []

for i in df1.ARREST_YEAR.unique():

    year_count.append(len(df1[df1['ARREST_YEAR']==i]))

plt.figure(figsize=(10,5))

sns.pointplot(x=df1.ARREST_YEAR.unique(),y=year_count,color='red',alpha=0.8)

plt.xlabel('Year',fontsize = 15,color='blue')

plt.xticks(rotation=45)

plt.ylabel('Crime Count',fontsize = 15,color='blue')

plt.title('Crime vs Year',fontsize = 15,color='blue')

plt.grid()

plt.show()

![image](https://user-images.githubusercontent.com/115191692/207205285-b7c5d201-c613-4bbb-974b-e70320e58363.png)

### So,From the above charts we can see that Category of Dangerous Drugs is most common crime among Age-group of 18-24 , Therefore we further analyse in the category of dangerous drugs.

df1=df1.loc[dt['OFNS_DESC'].isin(['DANGEROUS DRUGS'])]

df1.drop(df1.columns[[3]], axis=1, inplace=True)

![image](https://user-images.githubusercontent.com/115191692/207205920-fa9648c1-6801-4614-ac6b-4070b157089b.png)

# Reading CSV - Graduation Outcomes

import pandas as pd

import numpy as np

dt = pd.read_csv("bd1.csv")

dt

![image](https://user-images.githubusercontent.com/115191692/207206189-e9ab1c87-2911-4fe3-b7f9-01cede09e54b.png)

# Selecting data from a particular time period

dt1 =dt.loc[ dt['Cohort Year'] >= 2012]




##Dropping Unwanted Columns

dt1.drop(dt1.columns[[11,12,13,14,15,16,17,18,19,20,21]], axis=1, inplace=True)

dt1.drop(dt1.columns[[13,14,15,16]], axis=1, inplace=True)

dt2=dt1.loc[dt1['Demographic Category']=='All Students']

dt2

![image](https://user-images.githubusercontent.com/115191692/207208185-733cf317-8bd3-45ca-ac68-a05378eabccf.png)

### Correlation chart of Numeric Values

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

correlation_data=dt2.select_dtypes(include=[np.number]).corr()

mask = np.zeros_like(correlation_data, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))


cmap = sns.palette="vlag"

sns.heatmap(correlation_data, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5});


![image](https://user-images.githubusercontent.com/115191692/207208624-ca118f4f-fb28-4aa5-9633-c7aa4554c851.png)


### Correlation Heat Map

f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(dt2.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

![image](https://user-images.githubusercontent.com/115191692/207208948-673b2202-5147-4cfe-98d9-5cf8d4c65b31.png)


# Merging The datasets

grad=dt2

crime=df1

grad.rename(columns = {'Cohort Year':'Year'}, inplace = True)

crime.rename(columns = {'ARREST_YEAR':'Year'}, inplace = True)

final=pd.merge(grad,crime,on=['Year'])

final

![image](https://user-images.githubusercontent.com/115191692/207212241-1540df55-f069-4859-a861-36e8320468a5.png)

## Description Of Columns:
1. District: District/County number
2. Year: Year of Cohart/ Year of the arrest
3. Total Cohort #: No. of students who entered into university for 4 year degree
4. Total Grads #: No. of students Graduated
5. Total Grads of Cohort%: percentage of cohort who graduated
6. Total Regents #: No. of supervisors, who exercises general supervision over the conduct and welfare of the students.
7. Total Regents % of Cohort: percentage of supervisor among Cohort
8. Total Regents % of Grads:percentage of supervisor among Grads
9. Dropped out #: Total no. of students dropped out
10. Dropped out % of Cohort: Percentage of cohorts who dropped out
11. ARREST_DATE: Date of arrest
12. PD_DESC: Crime Description
13. PERP_SEX: Sex of perp who commited crime
14. PERP_RACE: Race of perp who commited crime
15. Weekday: Day of arrest
16. Day: Date of arrest
17. Month: Month of arrest

### Now, when the data is merged, the data can be visualized and lead us to conclusion.

# Conclusion

## Data Visualization:

### The proportion idea of Grads, Dropouts and regents:


![image](https://user-images.githubusercontent.com/115191692/207213398-edb18bf6-766e-4ca5-8cd6-e0e82e2ab39f.png)

### The Average count of arrests per day decreases as the no. of drop-outs decrease across the years, which shows both the variables are collinear 

![image](https://user-images.githubusercontent.com/115191692/207216285-63912bb1-b597-42cd-a4ad-641fde9da937.png)

![image](https://user-images.githubusercontent.com/115191692/207222018-4ab4f12e-eae8-4aeb-bfdd-9614c9054e1c.png)


### Sum of total no. of dropouts and count of crime description by arrest year and sex of prep

![image](https://user-images.githubusercontent.com/115191692/207216495-a0d63c69-ed15-4489-92ae-27f218b75abe.png)

### The Average count of arrests per day decreases as the no. of regents increase across the years, which shows both are inversely propotional to each other as the no. of superisors increases the crime decreases, as the student recieves proper advice and guidance for better future.

![image](https://user-images.githubusercontent.com/115191692/207216630-de6fe85f-b535-4154-9c6d-934e4e12fd9d.png)

### The Average count of arrests per day decreases as the no. of grads increase across the years, which shows both are inversely propotional to each other, the graduates get a proper scope for future, which leads decrease in crime.

![image](https://user-images.githubusercontent.com/115191692/207221904-da051b5a-f6ab-4b8d-a424-3dc8f5cd2c9b.png)








































