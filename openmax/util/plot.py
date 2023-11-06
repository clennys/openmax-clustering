import pickle
import matplotlib.pyplot as plt


with open('experiment_data/oscr_data_base_1_8_ORIGINAL.pkl', 'rb') as f:
    original = pickle.load(f)

with open('experiment_data/oscr_data_base_1_8_VALUE_SHIFT.pkl', 'rb') as f:
    val = pickle.load(f)

with open('experiment_data/oscr_data_base_1_8_ABS_REC_ACTV.pkl', 'rb') as f:
    abs = pickle.load(f)

fig, axs = plt.subplots(1, 1)
# key = '1200-1.25'
key = '1500-1.7'
axs.plot(original[key][1], original[key][0], label="original")
axs.plot(val[key][1], val[key][0], label="value-shift")
axs.plot(abs[key][1], abs[key][0], label="multpl")
plt.legend(loc="lower right")
plt.title(key)
plt.show()





