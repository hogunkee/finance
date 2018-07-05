import pandas as pd
import datetime
stock = pd.read_csv("stock.csv")
stock1 = stock[:200]
stock2 = stock[200:]

print(stock1.head(40))
print(stock2.head(40))
stock1.to_csv("valid_stock.csv", index=False)
stock2.to_csv("train_stock.csv", index=False)