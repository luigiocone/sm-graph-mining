import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(filepath_or_buffer="metrics.csv", delimiter=',')
df['file'] = df['file'].str[13:-4]
plt.scatter(df['file'], df['diameter'])
plt.xticks(rotation=90)
plt.show()