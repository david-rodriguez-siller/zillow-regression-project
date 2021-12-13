# Zillow Regression Project
### Project Objectives
 - Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook Final Report.

 - Create modules (acquire.py, prepare.py) that make your process repeateable and your report (notebook) easier to read and follow.

 - Ask exploratory questions of your data that will help you understand more about the attributes and drivers of home value. Answer questions through charts and statistical tests.

 - Construct a model to predict assessed home value for single family properties using regression techniques.

 - Make recommendations to a data science team about how to improve predictions.

 ### Project Goal
 The goal of this project is to predict assessed home values by the respective tax authority as accurately as possible based on a zillow dataset of home listings from 2017.

 ### Project Description
 Home value appraisals are a big factor in deciding which home to live in. A property's assessed tax value can be contested, but generally falls in a range assessed. That assessed value coult ultimately be a big factor on whether or not a prospective buyer purchases a home. Naturally, property taxes can be a big liability for homeowners and therefore it is an important factor that must be taken into account. Based on data from home listings from 2017, the goal of this exercise is to form a model to predict assessed home values.

 ### Initial Questions
 1. Given the information we have, what is the biggest factor of assessed taxes in homes?
2. Does the location of the home matter?
3. Is the total square feet of a home relevant at all in predicting assessed value?
4. Does the number of bedrooms or bathrooms relevant?

### Data Dictionary

| Variable  | Description |
| ------------- | ------------- |
| 1. bedroom_cnt | Number of bedrooms in each home. | 
| 2. bathroom_cnt | Number of bathrooms in each home. Can be half bathrooms and there will not be rows with 0 bathrooms |
| 3. pool_cnt | Number of pools in each home. |
| 4. nbr_stories | Number of stories in each home. |
| 5. assessed_tax_value | The target variable. Assessed property tax value of each home. |
| 6. year_built | The year the home was built. |
| 7. fips | Coding used to identify the county in which the home is located in. |
| 8. comb_sq_ft | An amalgamation of three different columns from the Zillow database. This column is a sum of columns: basementsqft, garagetotalsqft, and calculatedfinishedsquarefeet. |
| 9. location | A mapped fips column reflecting the name of the county of the fips code represents. |

### Steps to Reproduce
1. Read and understand this readme file.
2. Download companion .py files for access to functions defined and included in final report.
3. Have access to an env.py file with access to the codeup sql server in order to be able to pull the zillow dataset.
4. Follow each step as shown in the final report notebook.

### The plan

Initial Planning:
 - Understand how property taxes are assessed in California.
 - Make notes on what you expect to find in the dataset prior to exploring it.
 - Explore the dataset prior to importing to python to identify relevant columns that could be helpful in predicting the assessed home value.

Wrangle:
 - Import relevant modules into your python notebook.
 - Pull the data and relevant columns into Python.
 - Clean up the data, impute where necessary. Drop where necessary. Create new features if it makes sense at this stage.
 - Ensure that missing values are dropped or imputed.
 - Remove outliers and once the dataset is ready for exploration, begin exploring.

 Exploration: 
 - Identify your target variable.
 - Analyze each variable visually or by other tools available through python.
 - Move on to biviariate and multivariate exploration once initial exploratin has been exhausted.
 - Formulate and test hypotheses if deemed adequate.

 Modeling:
 - Prepare a baseline on the most adequate benchmark (mean, most, etc.)
 - Test different models on train and validate datasets.
 - Implement best performing model on test set.

