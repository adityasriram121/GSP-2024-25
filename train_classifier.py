import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

# Filter samples to ensure consistent feature length (21 landmarks x 2 = 42)
data_list = data_dict['data']
label_list = data_dict['labels']
filtered = [(x, y) for x, y in zip(data_list, label_list) if isinstance(x, (list, tuple)) and len(x) == 42]
if not filtered or len(filtered) < 2:
    raise ValueError(f'Not enough valid samples with 42 features: {len(filtered)}. Recreate dataset with clear hand images.')

data, labels = zip(*filtered)
data = np.asarray(data, dtype=float)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = accuracy_score(y_pred, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))


# Persist model along with learned class labels for correct inference mapping
class_names = list(model.classes_)
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'classes': class_names}, f)
f.close()


