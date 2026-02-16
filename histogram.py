import matplotlib.pyplot as plt
import pandas as pd

tags = ('Accuracy', 'Precision', 'Sensitivity', 'Specificity')
# insert some newlines in the tags to better fit into the plot
tags = [tag.replace(' (', '\n(') for tag in tags]
a = (0.9494, 0.9038, 0.8989, 0.9663)
b = (0.9345, 0.8724, 0.8691, 0.9564)
c = (0.9848, 0.9776, 0.9773, 0.9886)
d = (0.8331, 0.6656, 0.6663, 0.8888)
e = (0.8112, 0.6238, 0.6224, 0.8741)
f = (0.8100, 0.6250, 0.6201, 0.8734)
# create a dataframe
df = pd.DataFrame({"Naive": a, "ETS": b, "ETS: Three classes": c, "ARIMA": d, "LSTM": e, "RNN": f}, index=tags)
ax = df.plot.bar(rot=0, figsize=(12, 5))
ax.legend(bbox_to_anchor=(-0.1, 0.5))
plt.tight_layout() # fit labels etc. nicely into the plot
plt.grid(alpha=0.5)

plt.savefig('histogram.jpg', format='jpg')
