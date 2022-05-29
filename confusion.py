from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix#Fit the model
import numpy as np
import matplotlib.pyplot as plt

ar = [[281,367],[219,1000]]
ar = np.array(ar)
print(ar)

# cf_matrix = confusion_matrix(ar.get(0), ar.get(1))
# print(cf_matrix)

import seaborn as sns

ax = sns.heatmap(ar, annot=True, fmt='.0f',cmap='Blues')

ax.set_title('Confusion matrix of the test set\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()