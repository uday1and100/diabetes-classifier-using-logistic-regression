import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score,recall_score,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
gb=LogisticRegression()
a = pd.read_csv("diabetes.csv")
col = list(a.columns)
col.remove("Outcome")
for i in col:
  for j in range(0,len(a[i])):
    if a[i][j]==0:
      a[i][j]=a[i].mean()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
y = pd.DataFrame(a["Outcome"],columns=["Outcome"])
x = a.drop("Outcome",axis=1)
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=
x_res,y_res = smk.fit_sample(x,y)
x_res = pd.DataFrame(x_res,columns=[x.columns])
y_res = pd.DataFrame(y_res,columns=[y.columns])
scaler = StandardScaler()
x_res = scaler.fit_transform(x_res)
x_res = pd.DataFrame(x_res,columns=[x.columns])
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.2, random_state=42)
gb.fit(x_train,y_train)
y_new = gb.predict_proba(x_test)
y_new = y_new[:,1]
y_new = pd.DataFrame(y_new,columns=["probability"])
thresholds = [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8]
tps=[]
fps=[]
for i in thresholds:
  k_new = y_new.copy()
  for j in range(0,len(k_new.probability)):
    if k_new.probability[j]>i:
      k_new['probability'][j]=int(1)
    else:
      k_new["probability"][j]=int(0)
  tps.append(recall_score(y_test,k_new))
  fps.append(1-precision_score(y_test,k_new))
  print(accuracy_score(y_test,k_new))
  print(confusion_matrix(y_test,k_new))
print(tps)
plt.plot(fps,tps)
plt.scatter(fps,tps)
plt.ylabel("true positive rate")
plt.xlabel("false positive rate")
plt.title("roc curve")
plt.show()
tps.sort()
fps.sort()
for i in range(0,len(tps)):
  print(tps[i],fps[i])
