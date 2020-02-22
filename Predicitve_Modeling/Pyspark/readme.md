## End to End Predictive modeling pipeline in Pyspark ##

**What is End to End Predictive modeling pipeline?**

This is a tool that can take in a data with a binary target and can produce predictive modeling results using minimal interference. This tool is built in pyspark 2.4 and python 3.5

**What is its main purpose?**

This tool enables quick iterations of the predictive models with all the necessary stats to quicken the process of the model building. 

**Who can use it?**

It is a general purpose code that can be modified to individual's preference. If you are not into data science, please make sure to go through the results diligently before making any decisions.
I am open to colloborating with any data science enthusiasts to make this pipeline even better. I can be contacted at ramcharan.kakarla@okstate.edu




**What do I need to run this tool?**

1. Dataset in distributed environment
2. This version currently supports binary targets
3. tested on Pyspark 2.4 and Python 3.5



**Expected Inputs:**
1. Dataset name
2. target name
3. Variable patterns that you would like to exclude from modeling
4. Variables you would like to exclude from modeling
5. Acceptable threshold (number of levels) of categorical variable



**Expected Outputs:**

1. Summary Statistics

![Summary Statistics](https://github.com/ramchak1/DataScience/blob/master/Predicitve_Modeling/Pyspark/SummaryStats.png)

2. Top Variables

![Top Variables](https://github.com/ramchak1/DataScience/blob/master/Predicitve_Modeling/Pyspark/Top_Variables.png)

3. Performance metrics across 4 different algorithms 

![Test Metrics](https://github.com/ramchak1/DataScience/blob/master/Predicitve_Modeling/Pyspark/Test_Metrics.png)

![KS Metrics1](https://github.com/ramchak1/DataScience/blob/master/Predicitve_Modeling/Pyspark/ks_stats.png)

4. Excel sheet with all the above metrics


**Other FAQs:**


**How are missing values handled:**

Numeric with mean and categorical with mode. If there are any categorical variables with mode missing, tool overwrites with value 'Unknown value'

**Is this the best model:**

No, this is a baseline model. Idea is to give a quick glace of what is a ball park metric of what can be achieved


For any other questions/feedback please reach out to me at ramcharan.kakarla@okstate.edu
 
