#import libraries
import pandas as pd

#set file path
data_path = "llm/data/wordnet_from_web/USGW_v0.9.tab"
data_output_path = "llm/data/wordnet_from_web/USGW_v0.9.csv"

#read in file
df = pd.read_csv(data_path, sep="\t")

#print first rows
print(df.head())

#save to csv
df.to_csv(data_output_path, index=False)

#TODO: check that length is correct
