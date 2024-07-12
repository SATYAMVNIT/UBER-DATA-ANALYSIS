# UBER-DATA-ANALYSIS

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing Dataset
dataset = pd.read_csv("UberDataset.csv")
dataset.head()


# To find the shape of the dataset
dataset.shape


# To understand the data more deeply, we need to know about the null values count, datatype, etc. So for that we will use the below code.
dataset.info()


# Data Preprocessing
# As we understood that there are a lot of null values in PURPOSE column, so for that we will me filling the null values with a NOT keyword. You can try something else too.
dataset['PURPOSE'].fillna("NOT", inplace=True)


# Changing the START_DATE and END_DATE to the date_time format so that further it can be use to do analysis.
dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'], 
									errors='coerce')
dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'], 
									errors='coerce')


# Splitting the START_DATE to date and time column and then converting the time into four different categories i.e. Morning, Afternoon, Evening, Night
from datetime import datetime

dataset['date'] = pd.DatetimeIndex(dataset['START_DATE']).date
dataset['time'] = pd.DatetimeIndex(dataset['START_DATE']).hour

#changing into categories of day and night
dataset['day-night'] = pd.cut(x=dataset['time'],
							bins = [0,10,15,19,24],
							labels = ['Morning','Afternoon','Evening','Night'])


# Once we are done with creating new columns, we can now drop rows with null values.
dataset.dropna(inplace=True)


# It is also important to drop the duplicates rows from the dataset. To do that, refer the code below.
dataset.drop_duplicates(inplace=True)


# Data Visualization
# In this section, we will try to understand and compare all columns.

# Let’s start with checking the unique values in dataset of the columns with object datatype.
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

unique_values = {}
for col in object_cols:
unique_values[col] = dataset[col].unique().size
unique_values


# Now, we will be using matplotlib and seaborn library for countplot the CATEGORY and PURPOSE columns.
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.countplot(dataset['CATEGORY'])
plt.xticks(rotation=90)

plt.subplot(1,2,2)
sns.countplot(dataset['PURPOSE'])
plt.xticks(rotation=90)


# Let’s do the same for time column, here we will be using the time column which we have extracted above.
sns.countplot(dataset['day-night'])
plt.xticks(rotation=90)


# Now, we will be comparing the two different categories along with the PURPOSE of the user.
plt.figure(figsize=(15, 5))
sns.countplot(data=dataset, x='PURPOSE', hue='CATEGORY')
plt.xticks(rotation=90)
plt.show()


# As we have seen that CATEGORY and PURPOSE columns are two very important columns. So now we will be using OneHotEncoder to categories them.
from sklearn.preprocessing import OneHotEncoder
object_cols = ['CATEGORY', 'PURPOSE']
OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
OH_cols.index = dataset.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = dataset.drop(object_cols, axis=1)
dataset = pd.concat([df_final, OH_cols], axis=1)


# After that, we can now find the correlation between the columns using heatmap.
plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(), 
			cmap='BrBG', 
			fmt='.2f', 
			linewidths=2, 
			annot=True)


# Now, as we need to visualize the month data. This can we same as done before (for hours). 
dataset['MONTH'] = pd.DatetimeIndex(dataset['START_DATE']).month
month_label = {1.0: 'Jan', 2.0: 'Feb', 3.0: 'Mar', 4.0: 'April',
			5.0: 'May', 6.0: 'June', 7.0: 'July', 8.0: 'Aug',
			9.0: 'Sep', 10.0: 'Oct', 11.0: 'Nov', 12.0: 'Dec'}
dataset["MONTH"] = dataset.MONTH.map(month_label)

mon = dataset.MONTH.value_counts(sort=False)

# Month total rides count vs Month ride max count
df = pd.DataFrame({"MONTHS": mon.values,
				"VALUE COUNT": dataset.groupby('MONTH',
												sort=False)['MILES'].max()})

p = sns.lineplot(data=df)
p.set(xlabel="MONTHS", ylabel="VALUE COUNT")


# Visualization for days data.
dataset['DAY'] = dataset.START_DATE.dt.weekday
day_label = {
	0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thus', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}
dataset['DAY'] = dataset['DAY'].map(day_label)


day_label = dataset.DAY.value_counts()
sns.barplot(x=day_label.index, y=day_label);
plt.xlabel('DAY')
plt.ylabel('COUNT')


# Now, let’s explore the MILES Column .

# We can use boxplot to check the distribution of the column.
sns.boxplot(dataset['MILES'])


# As the graph is not clearly understandable. Let’s zoom in it for values lees than 100.
sns.boxplot(dataset[dataset['MILES']<100]['MILES'])


# It’s bit visible. But to get more clarity we can use distplot for values less than 40.
sns.distplot(dataset[dataset['MILES']<40]['MILES'])










