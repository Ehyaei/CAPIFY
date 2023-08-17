from utils import result_to_DF

# Load results
path = '../results/'
results = result_to_DF(path)
results.to_csv("../data/result_table.csv", index=False)