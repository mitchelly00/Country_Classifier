import pandas as pd
import boto3


df = pd.read_csv("combined_tables_with_validation.csv")



print(df['Found'].value_counts())
print("Removing not found")
print("")

df_found = df[df['Found']==True]

print(df["Dance"].value_counts())

print(sum(df_found["Dance"].value_counts().values))
