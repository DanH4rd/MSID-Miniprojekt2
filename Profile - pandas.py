import pandas_profiling
import pandas as pd


#df = pd.read_csv("Assessor_Historical_Secured_Property_Tax_Rolls.csv")
#df = pd.read_csv("./clean_data_without_nans.csv",sep=';', thousands=',')
df = pd.read_csv("./clean_data_without_nans_and_zeroes.csv")

profile = pandas_profiling.ProfileReport(df, correlations=None)

profile.to_file('filtered_data_profile_with_no_zeroes.html')
