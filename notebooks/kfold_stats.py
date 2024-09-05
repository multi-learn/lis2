import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 20})

with h5py.File("../../Results/nh2_dataset/naive_kfold_stats.h5", "r") as seg_perf:
    number_of_train_spine_pixels_nkf = np.array(seg_perf.get("number_of_train_spine_pixels"))
    number_of_test_spine_pixels_nkf = np.array(seg_perf.get("number_of_test_spine_pixels"))
with h5py.File("../../Results/nh2_dataset/random_kfold_stats.h5", "r") as seg_perf:
    number_of_train_spine_pixels_rkf = np.array(seg_perf.get("number_of_train_spine_pixels"))
    number_of_test_spine_pixels_rkf = np.array(seg_perf.get("number_of_test_spine_pixels"))
k = number_of_test_spine_pixels_rkf.shape[0]

G = gridspec.GridSpec(1, 2)
axes_1 = plt.subplot(G[0, 0])

axes_1.plot(np.linspace(0,k - 1, k), number_of_train_spine_pixels_rkf, label="random", marker="+")
axes_1.plot(np.linspace(0,k - 1, k), number_of_train_spine_pixels_nkf, label="naive", marker="*")
axes_1.set_title("Number of pixels labelled as filament in train")
axes_1.legend()

axes_2 = plt.subplot(G[0, 1])
axes_2.plot(np.linspace(0,k - 1, k), number_of_test_spine_pixels_rkf, label="random", marker="+")
axes_2.plot(np.linspace(0,k - 1, k), number_of_test_spine_pixels_nkf, label="naive", marker="*")
axes_2.set_title("Number of pixels labelled as filament in test")
axes_2.legend()
plt.show()