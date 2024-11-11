import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.DataFrame({
    'Milk': [1, 1, 1, 0, 1],
    'Bread': [1, 1, 0, 1, 1],
    'Butter': [1, 0, 1, 1, 1],
    'Beer': [0, 1, 1, 1, 1]
})
print("Original DataFrame:\n",df)

df_binary = data.astype(bool)
frequent_itemsets = apriori(df_binary, min_support=0.5, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6, num_itemsets=5)
print("\nAssociation Rules:")
print(rules)
