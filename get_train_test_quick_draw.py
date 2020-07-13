import pickle
import os
from tqdm import tqdm
import random

train_x = []

valid_x = []
test_x = []

root =  '/scratche/home/<user>/processed_quick_draw/'
for r, d, f in os.walk(root):
    for dr in tqdm(d):
        for r2,d2,f2 in os.walk(os.path.join(root, dr)):
            for file in tqdm(f2):
                if '.pkl' in file:
                    toss_1 = random.random()
                    if toss_1 > 0.8:
                        toss_2 = random.random()
                        if toss_2 > 0.5:
                            valid_x.append(os.path.join(os.path.join(root,dr), file))
                        else:
                            test_x.append(os.path.join(os.path.join(root,dr), file))
                    else:
                        train_x.append(os.path.join(os.path.join(root,dr), file))

print(len(train_x), len(test_x), len(valid_x))
data = {'train_x': train_x, 'test_x': test_x, 'valid_x': valid_x}
        print(positive_samples.size())
pickle.dump(data, open('processed_quick_draw_paths.pkl', 'wb'))
