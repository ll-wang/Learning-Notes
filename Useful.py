# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:40:03 2019

@author: wangl
"""
#ignor warning
import warnings
warnings.simplefilter('ignore')
#About parsing variables
df['new_col'] = df.var.str[0] #First character of the variable
df['new_col'] = df.var.str.split('-')
df['new_col'] = df.var.str.get(0) # first part of the list

'''np.array can do element-wise calculation'''

#.apply method
df['new_col'] = df.var.apply(len) #the new variable returns the length of the old one
df['new_col'] = df.var.apply(str.upper) #returns the upper case of the old variable

#Flexible arguments
def add_all(*args):
    sum_all = 0
    for num in args:
        sum_all += num
    return sum_all
# add_all(5,10,15,20)
    
def print_all(**kwargs): #key-value pairs
    for key,value in kwargs.items():
        print(key + \ ':\ ' + value )
# print_all(name = 'dumbledore', job = 'headmaster')

#Map and lambda function: map() applies the function to all elements ifn the sequence
nums = [48,6,9,21,1]
square_all = map(lambda x: x**2, nums)    

'''.map()method is used to transform values accroding the a dictionary look-up'''
red_vs_blue = {'Obama':'blue','Remoney':'Red'}
election['color'] = election.winner.map(red_vs_blues)

#Change dataframe to array
df_array = df.values

#Melting and pivot
'''1. Melting: id_vars: columns you don't want to melt; value_var: columns you want to melt; var_name:
    new name you provide to contain the melting column names; value_name: the name you provide of a new 
    column that contains the value of the melted columns'''
pd.melt(frame = df, id_vars = 'name', value_vars = ['t_a','t_b'], var_name = 'treatment', value_name = 'results')

'''2. Pivot: .pivot_table can do aggfunc but .pivot cannot
   index: columns you want to fix; columns: columns you want to pivot'''
weather_tidy = weather.pivot_table(index = 'date',columns = 'element', values = 'values', aggfunc = np.mean)

#Concatenate, Join, Merging
'''1. Concat'''
concatenated = pd.concat([var1,var2], ignor_index = True, axis = 0 '''append''' ''' or 1;merge''', join = 'inner' #'outer')
                         
'''Combine two columns'''
combined = df1.var1.str.cat(df1.var2, sep = ' ')                         

'''2. Merge data'''
pd.merge(left = data1, right = data2, on = None, left_on = 'df1_var1', right_on = 'df2_var2')  
pd.merge(df1,df2, on = ['var1','var2'], suffixes = ['_bronze','_gold'], how = 'inner' #left, right, outer) 
pd.merge_ordered(..., fill_method = 'ffill' #'bfill')         

'''3. Join method'''
df1.join(df2, how = 'inner' #left,right,outer)   

'''4. Combine columns as one'''
text_data.apply(lambda x: " ".join(x), axis=1)   #text_data includes all columns have text data                
                         
#Globbing: Read in all same type of files in a folder
import glob
csv_file = glob.glob('*.csv')
list_data = []
for filename in csv_file:
    data = pd.read_csv(filename)
    list_data.append(data)
pd.concat(list_data) #Combine all read in data together

#Convert data types of variables
df['var1'] = df.var1.astype(str) #change to str/int
df['var2'] = df.var2.astype('category')
df['var3'] = pd.to_numeric(df['var1'], errors = 'coerce') #change to numeric

#Redefine var as an ordered categorical
df['var'] = pd.Categorical(values = df.var,
                            categories = ['Bronze','Silver','Gold'],
                            ordered = True)
#Regular Expression


#Read in data
df = pd.read_csv(filepath, header = None, 
                           names = col_name, #List of column names to use
                           na_values = {'var1': ['-1']}, #change -1 values as NaN
                           parse_dates[[0,1,2]]#parse the date column into three columns
'''Slicing Time'''
df = pd.read_csv(filename, index_col = 'Date', parse_dates = True)  

#About date
'''Convert date_list into a datetime object'''
my_datetime = pd.to_datetime(date_list,format = '%y-%m-%d %H:%M') 

'''Convert string to datetime'''
import datetime

my_datetime = datetime.datetime.strptime('09-13-2019','%m-%d-%Y')

'''Resampling: using time series index'''
daily_mean = sales.resample('D').mean() #Table are be referred in P9-back
yearly = post2008.resample('A').last() #keeping last year date

'''Extract hour from a date column'''
sales['Date'].dt.hour

central = sales['Date'].dt.tz_localize('US/Central') 
central = sales['Date'].dt.tz_convert('US/Eastern')                       
                           
#Filtering
'''1. select rows contains specific values'''
indices = df['var1'] == 'str'
df2 = df.loc[indices,:]     

df2 = df.filter(lambda x: x['var'].sum()>35)
under10 = (titanic['age']<10).map({True:'under 10',False:'over 10'})  

# Filtering out rows without a market capitalization
cap = market_cap_raw.query('market_cap_usd > 0')

#Filtering out unuseful columns
for sub in ['^its_e', '^its_m', '^its_r', '^its_s']:
    var = par.filter(regex=sub, axis=1).fillna(0)  # Select e_vectors
#reindexing the index
ts4 = ts2.reindex(ts1.index, method = 'ffill') # or 'bfill'
ts2_interp = ts2.reindex(ts1.index).interpolation('linear')
#Interpolation
df.first().interpolate('linear') #.first() is in order to fill missing value if the first row is missing

#Add leading 0s
df['var1'] = df.var1.apply(lambda x: '{:0>4}'.format(x))#Add 0s so the total length is 4   

#.all() and .any(): .any() will return True if any of the element satisfy the condition; will return False if all elements does not satisfy
df2.loc[:,df.all()] #select columns with all nonzeros
df2.loc[:,df.any()]#select columns that not all observations are 0s
df.loc[:,df.notnull().any()] #select columns that not all observations are NaN
df.dropna(how = 'any') #drop rows if any of element is missing
df.dropna(thresh = 1000, axis = 'columns') #drop columns with < 1000 non-missing values
df.dropna(subset = ['stop_date','stop_time'],inplace = True) #drop rows

#MultiIndex
NY_month1 = sales.loc['NY',1] #'NY' is the outter row index and 1 is the inner row index

'''Access all the inner month index and look up data for all states in month2'''
all_month2 = sales.loc[(slice(None),2),:]

'''Look up data for CA and TX in month 2'''
CA_TX_month2 = sales.loc[(['CA','TX'],2),:]

'''Unstacking a multi-index'''
df.unstack(level = 1) #make long as wide
df.stack(lelve = 'gender') #Reverse of the above one

#Group by with customized aggregation
sales.groupby('customers')[['bread','butter']].agg({'bred':'sum','butter':data_range})
results = ab_test_results.groupby('group').agg({'uid':pd.Series.nunique}) #Calculate ncount for each group
#Rename
df.columns = df.columns.str.replace('F','C') #Rename 'F' in column names with 'C'
df.rename(columns = {'A':'B'})
#Multiply two columns in different dataframes with the same dates index
df_multiply = df1.multiply(df2['GBP/USD'], axis = 'rows')

#Delete duplicates and keep the most recent one
df2 = df2.sort_values('admin_time').drop_duplicates('Reg_ID', keep = 'last')
df = df.sort_values(['var1','var2'])
df = df.drop_duplicates(['var1','var2']) #Delete duplicates based on multiple variables

#keep duplicates
pull_name = df[['Last Name', 'First Name', 'TestCenter']].duplicated(keep=False)]
# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)

#Fill missing values with some other column values
df.var1.fillna(df.var2, inplace = True)

#Drop columns
df.drop(['var1','var2'], axis = 1, inplace = True)

#percentage change
yearly['growth'] = yearly.pct_change()*100

#Computing percentiles
np.percentile(df['var'],[25,50,75])
#Computing a frequency table
pd.crosstab(df.var1,df.var2)
#Exchange the level of column index
df2 = df.swaplevel(i = 2,j = 1,axis = 1)

#Select rows based on multiindex values
df2 = df[df.index.get_level_values('index_name') == 'values']

#Select variables in a list
df2 = df.loc[df.TestForm.isin(['19J','19L','19Q','19W'])]

#Calculate point biserial
def pbrs(series,subscore,par): 
    pbr = stats.pointbiserialr(series.fillna(0),par[subscore].fillna(0)) #function to calculate pbr
    return pbr
#Return the largest/smallest n rows by columns
df.nlargest(n, 'var')
df.nsmallest(n,'var')
#Create variables with a series postfix
pos = {}
for i in range(75):
    pos['Item_{}'.format(i+1)] = p_value
    
#os.path.join() method: can be used select multiple files in a folder
import os
'''Case 1'''
path = '/home/'
print(os.path.join(path, 'User/Desktop/','file.txt'))    

'''Case 2'''
path = "/User"

print(os.path.join(path,"Downloads","file.txt","/home"))

'''Case 3'''
path = "/User/Documents"

print(os.path.join(path,"/home/","file.txt"))

'''Case 4'''
path = '/home/'

print(os.path.join(path,'User/Public/','Documents/',''))

#Application
emails = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

#Return column name with max values of each row: idxmax()
df['new_colum'] = df.idxmax(axis = 1)

#Use column name to look up a column value
df['new_column'] = df.lookup(df.index, df.name) #df.name is the column contain the column names

#Select rows that the record is not an empty list
b = df[df.column.astype(bool)] #Python return the empty list as False 

#Plot with each column as a subplot
stock_data.plot(title = 'Stock Data', subplots = True)   
table.plot(kind = 'bar',stacked = True) #Stacking the bars
#save figure
plt.savefig('.png') # or .jpg or .pdf

#If there is an Nan and you want to split a string of a column
df['var'] = [x[0] if isinstance(x,str) else np.nan for x in df['var1']]

#Get the maximum length of a column
length = df['var'].astype(str).map(len).max()
#Plot distribution
sns.distplot(df['var'], fit = norm, hist = False)

#About scipy
from scipy.stats import pearsonr
corr,pvalue = pearsonr(var1,var2) #Calculate correlation and p-value testing non-correlation

#Plot the data on the Z-score scale and fit the line with least-square method
stats.probplot(df[feature], plot=plt)
df['A'].corr(df['B'])
#Show options
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows',500)
pd.set_option('display.width',100)     

#Set plot style
plt.style.use('ggplot')
sns.set_style('white') 
plt.figure() #so that a new figure can be set up each time  


#Generate multiple plots and save in one pdf file
from matplotlib.backends.backend_pdf import PdfPages

    def plots(g1, g2, var, path):
        plt.ioff()  # Turn off interactive mode to prevent figures showing on the screen
        out_pdf = r'{}'.format(path)
        pdf = PdfPages(out_pdf)
        sub_list = [('E', 'English'), ('M', 'Math'), ('R', 'Reading'), ('S', 'Science')]
        for vars in var:
            plot_num = 221
            fig = plt.figure()
            for sub in sub_list:
                print(plot_num)
                plt.subplot(plot_num)
                plt.plot(g1[sub[0]][vars])
                plt.plot(g2[sub[0]][vars])
                plt.xlabel('Item position')
                plt.ylabel(vars + ' (sec)')
                plt.legend(['longform', 'shortform'], loc='best')
                plt.title(vars + ' for ' + sub[1])
                plot_num += 1
            pdf.savefig(fig)
        pdf.close()

#Plots with different scales in a graph
def diff_plots(g1, g2, path):
    sub_list = [('E', 'English', 45), ('M', 'Math', 55), ('R', 'Reading', 40), ('S', 'Science', 40)]
    out_pdf = r'{}'.format(path)
    pdf = PdfPages(out_pdf)
    for sub in sub_list:
        g1common = g1[sub[0]].loc[g1[sub[0]].item_position.astype(int) <= sub[2]]
        pvalue_diff = g1common.p_value - g2[sub[0]].p_value
        avg_diff = g1[sub[0]].Avg_latency - g2[sub[0]].Avg_latency
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        l1 = ax1.plot(avg_diff, c='red', label='Avg_latency', linewidth=1)
        ax1.set_xlabel('Item position', fontsize=10)
        ax1.set_ylabel('Avg_latency_diff (longform - shortform)', fontsize=10)
        ax1.set_title('Common item difference between long form and short form for ' + sub[1], fontsize=12)

        ax2 = ax1.twinx()
        l2 = ax2.plot(pvalue_diff, c='blue', label='P_value', linewidth=1)
        ax2.set_ylabel('P_value_diff (longform - shortform)', fontsize=10)

        ls = l1 + l2
        labels = [l.get_label() for l in ls]
        ax1.legend(ls, labels, loc='best', fontsize=8)
        # plt.subplots_adjust(left = .1, right = .9, top = .9, bottom = .1)
        fig.tight_layout(pad=3)
        pdf.savefig(fig)
    pdf.close()
#About sns countplot and barplot
'''1. Countplot - plot count of each group'''
_ = sns.countplot(x = 'class',data = df)
_ = sns.countplot(x = 'class',hue = 'gender',data = df, palette = 'RdBu')

'''2. Barplot  - plot different groups based on the value of y'''
_ = sns.barplot(x = 'day',y = 'total_bill',data = df)
_ = sns.barplot( x = 'day', y = 'total_bill', hue = 'sex',data = df) 

#Heatmap
_ = sns.heatmap(df.corr(),square= True,cmap = 'RdYlGn',annot = True)

#Plot regression line
prediction_space = np.linspace(min(var1),max(var1)).reshape(-1,1)
plt.plot(prediction_space,y_pred,color = 'balck',linewitdth = 3)

#About ML
'''1. Cross validation'''
from sklearn.model_selection import cross_val_score
reg = LinearRression()
cv_score = cross_val_score(reg,X,y,cv = 5)

lasso_coef = lasso.fit(X,y).coef_

# Turn off interactive mode to prevent figures showing on the screen
plt.ioff()

'''Parameter tune for Ridge'''
alpha_space = np.logspace(-4,0,50)
for alpha in alpha_space:
    ridge.alpha = alpha
    
from sklearn.metrics import cofusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

'''AUC'''
from sklearn.metrics import roc_auc_score
y_pred_proba = logreg.predict_proba(X_test)[:,1] 
roc_auc_score(y_test,y_pre_proba)   
cv_scores = cross_val_score(logreg,X,y,cv = 5, scoring = 'roc_auc')

#Hyperparameter tuning
'''Grid Search'''
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1,50)}
knn_cv = GridSearchCV(knn,param_grid, cv = 5)
knn_cv.fit(X,y)
knn_cv.best_params_
knn_cv.best_estimator_
knn_cv.best_score_

#Get the name of hyperparameters
lr.get_params()

'''RandomizedSearchCV: only a fixed number of hyperparameter settings 
    is sampled from specified probability distributions'''
    
#About regular expression
'''Reverse a string'''
string2 = string1[::-1]

'''Split a string into a list of substrings'''
my_str = 'This string will be split'
my_str.split(sep = ' ',maxsplit = 2)
my_str.rsplit(sep = ' ',maxsplit = 2)

my_str2 = 'This string will be split\nin two'
my_str2.splitlines()    

'''Search and replace'''
my_string = "Where's a  Waldo?"
my_string.find('Waldo') #will return the least index
my_string.index('Waldo') #similar to find, but when the substring you want to 
                        #search does not exist, return error instead of -1
try:
    my_string.index('Wendy')
except ValueError:
    print('Not found')        

'''Formatting strings'''
tool = 'car'
goal = 'patterns'
print('{title} try to {aim}'.format(title = tool, aim = goal))

print('Only {0:f}% of the {1}'.format(.512,'data')) # 0 is index, f means float
print('Only {0:.2f}% of the {1}'.format(.512,'data')) #.2f means keep 2 decimal places

'''Formatting datetime'''
from datetime import datetime
print(datetime.now())
print("Today's date is {:%Y-%m-%d %H:%M}".format(datetime.now()))

'''f-strings: allowed conversion:
    1. !s (string conversion)
    2. !r (string containing a printable representation, i.e., with quotes
    3. !a (some that !r but escape the non-ASCII characters
    Format specifiers:
        1. e (scientific notation, e.g.,  5 10^3)
        2. d (digit, e.g., 4)
        3. f (float, e.g., 4.5353)'''
way = 'code'
method= 'how'
print(f'Practcing {way} and {method} to')
print(f'Practicing {way!r}')

number = 20.122345
print('The number {number: .2f}% of the data')

today = datetime.now()
print(f"Today's day is {today: %B %d, %Y}")

#Find if specific libray is installed


#Regrular expression
import re
'''1. Find all matches of a pattern: re.findall(r"regex",string)'''
re.findall(r"#movies", "Love #movies! I had fun of #movies")

'''2. split a tring: re.split(r'regex',string)'''
re.split(r"!","Nice! this is very good! hi.")

'''3. Replace one or many matches with a string: re.sub(r'regex',new,string)'''
re.sub(r'yellow','nice','I have a yellow car and a yellow house')

'''4. Repeated characters'''
password = 'password1234'
re.search(r'\s{8}\d{4}',password) 

'''Lazy search: add ? at the end of pattern'''

#Iterate rows of a dataframe: not efficient for big data
for idx, row in df.iterrows():
    print(idx + ': ' + row['capital'])
    df.loc[idx, 'newcol']= len(row['country'])#add a new column based on the value of column country

#Add a new column: efficient way
df['newcol'] = df['country'].apply(len)
df['newcol'] = df['country'].apply(str.upper)

'''Simulation'''
#Generate a random number from 0 to 1
np.random.seed(123)
np.random.rand()
np.random.random(size = 4) #draw 4 random number between 0 to 1
np.random.randint(4,8) #generate intergers from 4 to 7

#Sampling from a binomial distribution
np.random.binomial(4,.5, size = 10)

#Sampling from a Poisson distribution
np.random.poisson(6,size = 10000)

#Sampling from a exponential distribution
mean = np.mean(inter_times)
np.random.exponential(mean, size = 10000)

'''Open and read a text file'''
with open(' ') as file:
    file.readline()
    counts_dict = {}
    for i in range (1000): #Process the first 1000 rows
        line = file.readline().split(',') #Split the current line into a list when meet ','
        first_col = line[0]
        if first_col in counts_dict.keys():
            counts_dict[first_col] +=1
        else:
            counts_dict[first_col] = 1
    print(counts_dict)            
 #Read a text file 2
 with open('path','r') as file:
     file.read #or file.readline

file = open('path','r')
text = file.read()
file.close()     

#Importing flat files using NumPy
filename = ' '
data = np.loadtxt(filename,delimiter = ',')
#Optional arguments: skiprows = 1; usecols = [0,2]; dtype = str(all variables are strings); names = True(there is a header)

#Importing SAS files
from sas7bdat import SAS7BDAT
with SAS7BDAT(' ') as file:
    df_sas = file.to_data_frame()
    
#About read in excel file
x1 = pd.read_excel('',sheetname = None) #Read in all sheets
x2 = x1.keys() #get all sheet names
x1['Gender'].head() #Get the sheet named Gender

#Loading JSON file in Python
import json
with open('**.json','r') as json_file:
    json_data = json.load(json_file)
for k,v in json_data.items():
    print(k + ': ',v)   
    
#Change the df.info() into dataframe
df_info = df_all.info()
import io
buffer = io.StringIO()
df_all.info(buf=buffer)
s = buffer.getvalue()

df_try= pd.read_table('df_info.txt',delim_whitespace = True, names = ('A','B','C'),
                      dtype= {'A':np.int64, 'B': np.float64, 'C': np.float64})
with open("df_info.xlsx", "w", encoding="utf-8") as f:
    f.write(s)

#2. Pivot table: un-melting data

'''About Visulization'''
    
    
    
          

                