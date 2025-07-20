# find average of duplicate students
import pandas as pd

df = pd.read_json("sample.json")

# false flag return all true duplicates
duplicate_names = df[df.duplicated(subset="name", keep=False)]

# grouping then applying aggregation
average_marks = duplicate_names.groupby("name")["marks"].mean().reset_index()

print(average_marks)
