##########################################################
# End to End Predictive modeling in Pyspark
##########################################################


# --Author     : Ramcharan Kakarla
# --Date       : 02/22/2020
# --Version    : 1
# --Environ    : Pyspark (2.4 & above) and python 3.5
# --Tested     : Passed in databricks community version
# --Dataset    : Churn Modeling `https://www.kaggle.com/shrutimechlearn/churn-modelling#Churn_Modelling.csv`
# --Objective  : Predict the likelihood of customer leaving the bank with avialable input data
# --Inputs     : Structured Data (without text fields), Binary Target
# --Outputs    : 1. Summary Statistics 2. Top Variables 3. Performance metrics across 4 algorithms
# --Validation : KS, Accuracy, Precision, Recall, F1 score
# --Feedback   : ramcharan.kakarla@okstate.edu





from pyspark import SparkContext,HiveContext,Row,SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import HiveContext
from pyspark.sql import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.sql.functions import *
from pyspark.mllib.stat import *
from pyspark.ml.feature import *
from pyspark.sql.types import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from sklearn.externals import joblib
from pyspark.ml.feature import IndexToString,StringIndexer,VectorIndexer,OneHotEncoderEstimator,VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
from pyspark.sql.functions import *
from pyspark.sql import functions as F
import builtins
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import random
import re


#Input table name
input_table='Churn_Modelling_csv'
#Target Variable name
target_variable='Exited'

#Variable filters
custom_filter='Y'
# This search term gives the ability to exclude variables by pattern
search_term=['_ID']
# These are the variables that can be excluded in the modeling exercise
exclude_list=['Surname']
#Maximum level of distinct levels allowable for categorical values
distinct_threshold=20

#Reading the table
df=spark.sql("select * from "+input_table)

#Temporary holder for the variables
newlist_temp=[]
names_var=df.columns
#This piece of code helps in filtering the unecessary variables from your dataset
search_term=[x.lower() for x in search_term]
if custom_filter=='N':
 regular_expression_search_to_filter=['nothing_to_filter']
elif custom_filter=='Y':
 regular_expression_search_to_filter=search_term
# We are iterating through the list of patterns provided to weed out unecessary varaibles
for len_list in range(0,len(regular_expression_search_to_filter)):
    r = re.compile(".*"+regular_expression_search_to_filter[len_list])
    newlist_ext = filter(r.match, names_var)
    newlist_temp.append(newlist_ext)
newlist=[item for sublist in newlist_temp for item in sublist]
exclude_list=[x.lower() for x in exclude_list]
#Preprocessing
# input_table=input_table.lower()
# target_variable=target_variable.lower()
#Variable List
v_list=[]
vars=df.dtypes

# This will ensure we are filtering date and timestamp variables from the dataset as they are not useful in their raw format
def char_num(data):
    vars=data.dtypes
    categorical=[]
    numeric=[]
    for i in range(0,len(vars)):
      if vars[i][1]=="string":
        categorical.append(vars[i][0])
      elif vars[i][1] not in ['timestamp','date']:
        numeric.append(vars[i][0])
    return categorical,numeric


categorical,numeric=char_num(df)
v_list=list(set(numeric)|set(categorical))
filter_list=list(set(newlist)|set(exclude_list))
final_list_vars=list(set(v_list)-set(filter_list))
# Filter the dataset
df=df.select(*final_list_vars)
vars_processed=df.columns
cats,nums=char_num(df)
print(' Total number of columns in the dataset is '+str(len(names_var))+'. The number of categorical columns is '+str(len(categorical))+' and numeric columns is '+str(len(numeric))+'.\n The number of valid columns in your dataset ' +input_table+ ' are ' +str(len(final_list_vars)) +' out of ' +str(len(names_var))+'.\n Among '+str(len(final_list_vars))+' variables, the number of categorical variables are '+str(len(cats))+' and numerical are '+str(len(nums)) )

print('Starting Summary_Stats')

def data_stats(df):
    #Missing percentages among the columns
    rows = df.count()
    summary = df.describe().filter(col("summary") == "count").drop('summary')
    df_total_counts=summary.toPandas().transpose()
    df_total_counts = df_total_counts.rename_axis('Variable').reset_index()
    df_total_counts.rename(columns={0:"Non_Missing"},inplace=True)
    df_missing_counts=summary.select(*((lit(rows)-col(c)).alias(c) for c in df.columns)).toPandas().transpose()
    df_missing_counts = df_missing_counts.rename_axis('Variable').reset_index()
    df_missing_counts.rename(columns={0:"Missing"},inplace=True)
    df_missing=df_total_counts.merge(df_missing_counts, on='Variable', how='left')
    df_missing['total_rows']=rows
    df_missing['missing_percent']=(df_missing['Missing']/df_missing['total_rows'])*100
    #Mode Calculation
    mode_vars=[[i,df.groupby(i).count().orderBy("count", ascending=False).first()[0]] for i in cats]
    ModeDict = {item[0]: item[1] for item in mode_vars}
    mode_values=pd.DataFrame([ModeDict]).transpose()
    mode_values = mode_values.rename_axis('Variable').reset_index()
    mode_values.rename(columns={0:"Mode"},inplace=True)
    #Mean Calculation
    mean_v = df.describe().filter(col("summary") == "mean").drop('summary')
    mean_values=mean_v.toPandas().transpose()
    mean_values = mean_values.rename_axis('Variable').reset_index()
    mean_values.rename(columns={0:"Mean"},inplace=True)
    df_stats=df_missing.merge(mean_values,on='Variable', how='left')
    df_stats=df_stats.merge(mode_values,on='Variable', how='left')
    #Variable datatypes
    datatype={item[0]: item[1] for item in df.dtypes}
    variable_dtypes=pd.DataFrame([datatype]).transpose()
    variable_dtypes = variable_dtypes.rename_axis('Variable').reset_index()
    variable_dtypes.rename(columns={0:"dtype_vars"},inplace=True)
    df_stats=df_stats.merge(variable_dtypes,on='Variable', how='left')
    #Distinct Values
    distinct_vals=df.agg(*(countDistinct(col(c)).alias(c) for c in cats)).toPandas().transpose()
    df_distinct_counts = distinct_vals.rename_axis('Variable').reset_index()
    df_distinct_counts.rename(columns={0:"distinct_values"},inplace=True)
    df_stats=df_stats.merge(df_distinct_counts,on='Variable', how='left')
    return df_stats

df_stats=data_stats(df)
print('Completed Summary_Stats')



#Mode Imputation for categoricals
df_mode=df_stats.loc[(df_stats['dtype_vars']=='string') & (df_stats['Mode'].notnull())][['Variable','Mode']]
mode_dictonary=df_mode.set_index('Variable').to_dict()

#Mean Imputation for numerics
df_mean=df_stats.loc[(df_stats['dtype_vars']!='string') & (df_stats['Mean'].notnull())][['Variable','Mean']]
mean_dictonary=df_mean.set_index('Variable').to_dict()


df = df.na.fill(mode_dictonary['Mode'])
df = df.na.fill(mean_dictonary['Mean'])

# If the missing value is the mode for categoricals, we will impute with unknowm
df = df.na.fill('Unknown_Value')


df_distinct_subset=df_stats.loc[(df_stats['distinct_values']>=distinct_threshold)& (df_stats['distinct_values'].notnull())]
vars_to_drop=df_distinct_subset['Variable'].tolist()
df=df.drop(*vars_to_drop)
df.cache()
cats=list(set(cats) - set(vars_to_drop))



stages=[]
label_indexer = StringIndexer(inputCol = ''+target_variable,outputCol = 'label').fit(df)
df=label_indexer.transform(df)
string_indexer= [StringIndexer(inputCol=column,outputCol=column+"_index",handleInvalid="keep") for column in cats]
pipeline0 = Pipeline(stages = string_indexer)
df_transformed=pipeline0.fit(df).transform(df)
df_transformed.cache()

#One hot encoding for Logistic
encoder = OneHotEncoderEstimator(inputCols=[string_indexer[i].getOutputCol() for i in range(0,len(string_indexer))], outputCols=[column + "_cat" for column in cats])
stages += [ encoder]
assemblerInputs = [c + "_cat" for c in cats] + nums
assemblerInputs.remove(target_variable)
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
pipeline1=Pipeline(stages = stages)
df_transformed_logistic=pipeline1.fit(df_transformed).transform(df_transformed)
standardscaler=StandardScaler().setInputCol("features").setOutputCol("scaled_features")
df_transformed_logistic=standardscaler.fit(df_transformed_logistic).transform(df_transformed_logistic)
train, test = df_transformed_logistic.randomSplit([0.70, 0.30], seed = 42)

print('Completed Preprocessing')




def train_test(model_to_implemented,train,test):
    predict_train=model_to_implemented.transform(train)
    predict_test=model_to_implemented.transform(test)
    split1_udf = udf(lambda value: value[1].item(), DoubleType())
    predict_train = predict_train.withColumn('model_probability', split1_udf('probability'))
    predict_test = predict_test.withColumn('model_probability', split1_udf('probability'))
    cols_req=[target_variable,'model_probability','prediction','rawPrediction','label']
    predict_train=predict_train.select(*cols_req)
    predict_test=predict_test.select(*cols_req)
    return predict_train,predict_test

def ks_func(predictions,y,probability):
    decileDF = predictions.select(y, probability)
    decileDF=decileDF.withColumn('non_target',1-decileDF[y])
    window = Window.orderBy(desc(probability))
    decileDF = decileDF.withColumn("rownum", F.row_number().over(window))
    decileDF.cache()
    decileDF=decileDF.withColumn("rownum",decileDF["rownum"].cast("double"))
    window2 = Window.orderBy("rownum")
    RFbucketedData=decileDF.withColumn("deciles", F.ntile(10).over(window2))
    RFbucketedData = RFbucketedData.withColumn('deciles',RFbucketedData['deciles'].cast("int"))
    RFbucketedData.cache()
    ## to pandas from here
    print('KS calculation starting')
    target_cnt=RFbucketedData.groupBy('deciles').agg(F.sum(y).alias('target')).toPandas()
    non_target_cnt=RFbucketedData.groupBy('deciles').agg(F.sum("non_target").alias('non_target')).toPandas()
    overall_cnt=RFbucketedData.groupBy('deciles').count().alias('Total').toPandas()
    overall_cnt = overall_cnt.merge(target_cnt,on='deciles',how='inner').merge(non_target_cnt,on='deciles',how='inner')
    overall_cnt=overall_cnt.sort_values(by='deciles',ascending=True)
    overall_cnt['Pct_target']=(overall_cnt['target']/overall_cnt['count'])*100
    overall_cnt['cum_target'] = overall_cnt.target.cumsum()
    overall_cnt['cum_non_target'] = overall_cnt.non_target.cumsum()
    overall_cnt['%Dist_Target'] = (overall_cnt['cum_target'] / overall_cnt.target.sum())*100
    overall_cnt['%Dist_non_Target'] = (overall_cnt['cum_non_target'] / overall_cnt.non_target.sum())*100
    overall_cnt['spread'] = builtins.abs(overall_cnt['%Dist_Target']-overall_cnt['%Dist_non_Target'])
    decile_table=overall_cnt.round(2)
    print("KS_Value =", builtins.round(overall_cnt.spread.max(),2))
    decileDF.unpersist()
    RFbucketedData.unpersist()
    return builtins.round(overall_cnt.spread.max(),2), overall_cnt

def validation_met(train_dataset,target_variable,prediction,rawPrediction,label):
    global_accuracy=train_dataset.select(target_variable,prediction).crosstab(target_variable,prediction).toPandas()
    global_accuracy.rename(columns={'0.0':"Predicted_O",'1.0':"Predicted_1"},inplace=True)
    global_accuracy.columns
    name_of_crosstab=target_variable+'_prediction'
    global_accuracy = global_accuracy.sort_values(by=[name_of_crosstab], ascending=True)
    True_Positive=global_accuracy.loc[global_accuracy[name_of_crosstab]=='1']['Predicted_1'][0]
    False_Positive=global_accuracy.loc[global_accuracy[name_of_crosstab]=='0']['Predicted_1'][1]
    True_Negative=global_accuracy.loc[global_accuracy[name_of_crosstab]=='0']['Predicted_O'][1]
    False_Negative=global_accuracy.loc[global_accuracy[name_of_crosstab]=='1']['Predicted_O'][0]
    Precision=float(True_Positive)/float(True_Positive+False_Positive)
    Recall=float(True_Positive)/float(True_Positive+False_Negative)
    F1_Score=2*(Precision*Recall)/(Precision+Recall)
    Accuracy=float(True_Positive+True_Negative)/float(True_Positive+True_Negative+False_Positive+False_Negative)
    evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='label')
    roc=evaluator.evaluate(train_dataset)
    return Precision,Recall,F1_Score,Accuracy,roc

#Logistic Regression
LR = LogisticRegression(featuresCol ='scaled_features', labelCol = 'label', maxIter=15)
print('Running Logistic')
LR_model = LR.fit(train)

predict_train,predict_test=train_test(LR_model,train,test)
log_train_precision,log_train_recall,log_train_f1,log_train_accuracy,log_train_roc=validation_met(predict_train,target_variable,'prediction','rawPrediction','label')
log_test_precision,log_test_recall,log_test_f1,log_test_accuracy,log_test_roc=validation_met(predict_test,target_variable,'prediction','rawPrediction','label')
logistic_train_ks,logistic_train_ks_tab=ks_func(predict_train,target_variable,'model_probability')
logistic_test_ks,logistic_test_ks_tab=ks_func(predict_test,target_variable,'model_probability')
logistic_train_ks_tab['type_of_model']='Logistic Train'
logistic_test_ks_tab['type_of_model']='Logistic Test'
logistic_train_ks_tab['ks']=logistic_train_ks
logistic_test_ks_tab['ks']=logistic_test_ks


#For Tree & mlp models
stages_tree=[]
TreeInputs = [c + "_index" for c in cats] + nums
TreeInputs.remove(target_variable)
assembler = VectorAssembler(inputCols=TreeInputs, outputCol="features")
stages_tree += [assembler]
pipeline1=Pipeline(stages = stages_tree)
df_transformed_trees=pipeline1.fit(df_transformed).transform(df_transformed)
standardscaler=StandardScaler().setInputCol("features").setOutputCol("scaled_features")
df_transformed_trees=standardscaler.fit(df_transformed_trees).transform(df_transformed_trees)
train, test = df_transformed_trees.randomSplit([0.70, 0.30], seed = 42)



# MLP
layers = [len(TreeInputs), len(TreeInputs)*3, len(TreeInputs)*2, 2]
mlp = MultilayerPerceptronClassifier(featuresCol = 'scaled_features', labelCol = 'label', maxIter=50, layers=layers, blockSize=512,seed=91)
print('Running MultiLayerPerceptron')
mlpModel = mlp.fit(train)

predict_train,predict_test=train_test(mlpModel,train,test)
predict_train.cache()
predict_test.cache()
mlp_train_precision,mlp_train_recall,mlp_train_f1,mlp_train_accuracy,mlp_train_roc=validation_met(predict_train,target_variable,'prediction','rawPrediction','label')
mlp_test_precision,mlp_test_recall,mlp_test_f1,mlp_test_accuracy,mlp_test_roc=validation_met(predict_test,target_variable,'prediction','rawPrediction','label')
mlp_train_ks,mlp_train_ks_tab=ks_func(predict_train,target_variable,'model_probability')
mlp_test_ks,mlp_test_ks_tab=ks_func(predict_test,target_variable,'model_probability')
mlp_train_ks_tab['type_of_model']='MLP Train'
mlp_test_ks_tab['type_of_model']='MLP Test'
mlp_train_ks_tab['ks']=mlp_train_ks
mlp_test_ks_tab['ks']=mlp_test_ks
predict_train.unpersist()
predict_test.unpersist()



#RandomForest
stages_tree=[]
classifier = RandomForestClassifier(labelCol = 'label',featuresCol = 'features',maxBins=120)
stages_tree += [classifier]
pipeline_tree=Pipeline(stages=stages_tree)
print('Running RFModel')
RFmodel = pipeline_tree.fit(train)
rfModel = RFmodel.stages[0]
Variables=list(rfModel.featureImportances)
vars_in_tree=TreeInputs
Top_Variables = pd.DataFrame({'Variable': vars_in_tree,'Variable_Importance': Variables})
Top_Variables=Top_Variables.sort_values(['Variable_Importance'],ascending=False)


predict_train,predict_test=train_test(RFmodel,train,test)
predict_train.cache()
predict_test.cache()
rf_train_precision,rf_train_recall,rf_train_f1,rf_train_accuracy,rf_train_roc=validation_met(predict_train,target_variable,'prediction','rawPrediction','label')
rf_test_precision,rf_test_recall,rf_test_f1,rf_test_accuracy,rf_test_roc=validation_met(predict_test,target_variable,'prediction','rawPrediction','label')
rf_train_ks,rf_train_ks_tab=ks_func(predict_train,target_variable,'model_probability')
rf_test_ks,rf_test_ks_tab=ks_func(predict_test,target_variable,'model_probability')
rf_train_ks_tab['type_of_model']='RF Train'
rf_test_ks_tab['type_of_model']='RF Test'
rf_train_ks_tab['ks']=rf_train_ks
rf_test_ks_tab['ks']=rf_test_ks
predict_train.unpersist()
predict_test.unpersist()



#Gradient Boosting
stages_tree_gbt=[]
gbt = GBTClassifier(featuresCol = 'features', labelCol = 'label',maxIter=15)
stages_tree_gbt += [gbt]
pipeline_tree_gbt=Pipeline(stages=stages_tree_gbt)
print('Running GBT')
GBT_Model = pipeline_tree_gbt.fit(train)


predict_train,predict_test=train_test(GBT_Model,train,test)
predict_train.cache()
predict_test.cache()
GBT_train_precision,GBT_train_recall,GBT_train_f1,GBT_train_accuracy,GBT_train_roc=validation_met(predict_train,target_variable,'prediction','rawPrediction','label')
GBT_test_precision,GBT_test_recall,GBT_test_f1,GBT_test_accuracy,GBT_test_roc=validation_met(predict_test,target_variable,'prediction','rawPrediction','label')
GBT_train_ks,GBT_train_ks_tab=ks_func(predict_train,target_variable,'model_probability')
GBT_test_ks,GBT_test_ks_tab=ks_func(predict_test,target_variable,'model_probability')
GBT_train_ks_tab['type_of_model']='GBT Train'
GBT_test_ks_tab['type_of_model']='GBT Test'
GBT_train_ks_tab['ks']=GBT_train_ks
GBT_test_ks_tab['ks']=GBT_test_ks

predict_train.unpersist()
predict_test.unpersist()



# Collating all the results :
print('Collecting all results')
Algorithms=['Logistic','MultiLayerPerceptron','RandomForest','GradientBoosting']
Train_Precisions=[log_train_precision,mlp_train_precision,rf_train_precision,GBT_train_precision]
Test_Precisions=[log_test_precision,mlp_test_precision,rf_test_precision,GBT_test_precision]
Train_Recall=[log_train_recall,mlp_train_recall,rf_train_recall,GBT_train_recall]
Test_Recall=[log_test_recall,mlp_test_recall,rf_test_recall,GBT_test_recall]
Train_F1=[log_train_f1,mlp_train_f1,rf_train_f1,GBT_train_f1]
Test_F1=[log_test_f1,mlp_test_f1,rf_test_f1,GBT_test_f1]
Train_Accuracy=[log_train_accuracy,mlp_train_accuracy,rf_train_accuracy,GBT_train_accuracy]
Test_Accuracy=[log_test_accuracy,mlp_test_accuracy,rf_test_accuracy,GBT_test_accuracy]
Train_ROC=[log_train_roc,mlp_train_roc,rf_train_roc,GBT_train_roc]
Test_ROC=[log_test_roc,mlp_test_roc,rf_test_roc,GBT_test_roc]
Train_KS=[logistic_train_ks,mlp_train_ks,rf_train_ks,GBT_train_ks]
Test_KS=[logistic_test_ks,mlp_test_ks,rf_test_ks,GBT_test_ks]

Other_Stats = pd.DataFrame(
    {'Algorithms': Algorithms,
     'Train_Precision': Train_Precisions,
     'Test_Precision': Test_Precisions,
     'Train_Recall': Train_Recall,
     'Test_Recall': Test_Recall,
     'Train_F1': Train_F1,
     'Test_F1': Test_F1,
     'Train_Accuracy': Train_Accuracy,
     'Test_Accuracy': Test_Accuracy,
     'Train_ROC': Train_ROC,
     'Test_ROC': Test_ROC,
     'Train_KS': Train_KS,
     'Test_KS': Test_KS
    })

Other_Stats=Other_Stats[['Algorithms','Train_Precision','Test_Precision','Train_Recall','Test_Recall','Train_F1','Test_F1','Train_Accuracy','Test_Accuracy','Train_ROC','Test_ROC','Train_KS','Test_KS']]

ks_all=pd.concat([logistic_train_ks_tab,logistic_test_ks_tab,mlp_train_ks_tab,mlp_test_ks_tab,rf_train_ks_tab,rf_test_ks_tab, GBT_train_ks_tab,GBT_test_ks_tab])


path = r"/FileStore/tables/test_metrics.xlsx"
writer = pd.ExcelWriter(path, engine = 'xlsxwriter')
df_stats.to_excel(writer,sheet_name='Summary_Stats',index=False)
Top_Variables.to_excel(writer,sheet_name='Top_Vars',index=False)
Other_Stats.to_excel(writer,sheet_name='Performance_metrics',index=False)
ks_all.to_excel(writer,sheet_name='KS',index=False)
writer.save()
writer.close()
