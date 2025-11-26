import numpy as np
import matplotlib.pyplot as plt

print("\n\n")
print("90\% HRL(only_subtask_termination) _ d, no downsampled\n")
print("---------------\n")

# Mean values
TN, TP = 0.94636, 0.05102
FP, FN = 0.00237, 0.00025

# Uncertainties
TN_err, TP_err = 0.00436, 0.00555
FP_err, FN_err = 0.00009, 0.00347

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")

print("\n\n")
print("100\% BORDER(only_subtask_termination) _ d, no downsampled\n")
print("---------------\n")

# Mean values
TN, TP = 0.94588, 0.04966
FP, FN = 0.00246, 0.00200

# Uncertainties
TN_err, TP_err = 0.00404, 0.01053
FP_err, FN_err = 0.00038, 0.01008

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")

print("\n\n")
print("100\% PLUME(only_subtask_termination) _  d, no downsampled\n")
print("---------------\n")

# Mean values
TN, TP = 0.94627, 0.05065
FP, FN = 0.00237, 0.00071

# Uncertainties
TN_err, TP_err = 0.00439, 0.00740
FP_err, FN_err = 0.00009, 0.00562

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")


print("\n\n")
print("100\% BORDER(only_subtask_termination) _ downsampled, no d\n")
print("---------------\n")

# Mean values
TN, TP = 0.94598, 0.05120
FP, FN = 0.00240, 0.00042

# Uncertainties
TN_err, TP_err = 0.00396, 0.00589
FP_err, FN_err = 0.00019, 0.00426

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")

print("\n\n")
print("100\% PLUME(only_subtask_termination) _ downsampled, no d\n")
print("---------------\n")

# Mean values
TN, TP = 0.94623, 0.05111
FP, FN = 0.00237, 0.00028

# Uncertainties
TN_err, TP_err = 0.00412, 0.00546
FP_err, FN_err = 0.00017, 0.00353

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")


print("\n\n")
print("100\% HRL _ downsampled no d\n")
print("---------------\n")

# Mean values
TN, TP = 0.94610, 0.05015
FP, FN = 0.00238, 0.00137

# Uncertainties
TN_err, TP_err = 0.00412, 0.00882
FP_err, FN_err = 0.00015, 0.00797

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")

print("\n\n")
print("100\% PLUME _ d_no_downsampled\n")
print("---------------\n")

# Mean values
TN, TP = 0.94602, 0.05153
FP, FN = 0.00237, 0.00007

# Uncertainties
TN_err, TP_err = 0.00390, 0.00423
FP_err, FN_err = 0.00009, 0.00157

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")
print("\n\n")
print("100\% BORDER_ d_no_downsampled\n")
print("---------------\n")

# Mean values
TN, TP = 0.94591, 0.05100
FP, FN = 0.00238, 0.00071
# Uncertainties
TN_err, TP_err = 0.00398, 0.00688
FP_err, FN_err = 0.00012, 0.00562

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")

print("\n\n")
print("100\% PLUME _ downsampled_no_d\n")
print("---------------\n")

# Mean values
TN, TP = 0.94612, 0.05048
FP, FN = 0.00237, 0.00103

# Uncertainties
TN_err, TP_err = 0.00402, 0.00798
FP_err, FN_err = 0.00009, 0.00703

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")
print("\n\n")
print("100\% BORDER_ downsampled_no_d\n")
print("---------------\n")

# Mean values
TN, TP = 0.94591, 0.05136
FP, FN = 0.00240, 0.00033

# Uncertainties
TN_err, TP_err = 0.00399, 0.00496
FP_err, FN_err = 0.00033, 0.00300

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")

print("\n\n")
print("90\% HRL _ d_no_downsampled\n")
print("---------------\n")

# Mean values
TN, TP = 0.94613, 0.05147
FP, FN = 0.00237, 0.00003

# Uncertainties
TN_err, TP_err = 0.00387, 0.00404
FP_err, FN_err = 0.00007, 0.00081

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")

print("\n\n")
print("100\% HRL _ d_no_downsampled\n")
print("---------------\n")

# Mean values
TN, TP = 0.94581, 0.05170
FP, FN = 0.00237, 0.00011

# Uncertainties
TN_err, TP_err = 0.00365, 0.00422
FP_err, FN_err = 0.00008, 0.00202

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")

print("\n\n")
print("mean unc-lawnmower\n")
print("---------------\n")

# Mean values
TN, TP = 0.94884, 0.04878
FP, FN = 0.00238, 0.00000

# Uncertainties
TN_err, TP_err = 0.00595, 0.00595
FP_err, FN_err = 0.00000, 0.00000

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")


# Mean values
TN, TP = 0.94630, 0.05022
FP, FN = 0.00280, 0.00039

# Uncertainties
TN_err, TP_err = 0.0444, 0.00708
FP_err, FN_err = 0.00009, 0.00417

print("LAWNMOWER PERFORANCE\n")
print("---------------\n")
print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}") # out of all positive, how many of them did we find?


print(f"Specitivity={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}") # how many negative cells did we correctly identify?

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}") # how many were positive of the ones considered positive?
# Confusion matrix (normalized)
print("\n\n")
print("VAR reduction PERFORANCE\n")
print("---------------\n")

# Mean values
TN, TP = 0.94561, 0.05119
FP, FN = 0.00280, 0.00039

# Uncertainties
TN_err, TP_err = 0.0409, 0.00517
FP_err, FN_err = 0.00069, 0.00332

print(f"Recall={(TP)/(TP+FN):.5f} ± {np.sqrt((TP_err/(TP+FN) - (TP)/(TP+FN)**2*TP_err)**2 + (-TP*FN_err/(TP+FN)**2)**2):.5f}")


print(f"TPN={(TN)/(TN+FP):.5f} ± {np.sqrt((TN_err/(TN+FP) - (TN)/(TN+FP)**2*TN_err)**2 + (-TN*FP_err/(TN+FP)**2)**2):.5f}")

print(f"Precision={(TP)/(TP+FP):.5f} ± {np.sqrt((TP_err/(TP+FP) - (TP)/(TP+FP)**2*TP_err)**2 + (-TP*FP_err/(TP+FP)**2)**2):.5f}")

import matplotlib.pyplot as plt
import numpy as np

metrics = ['Recall', 'TNR', 'Precision']
values = np.array([0.99229, 0.99705, 0.94719])
errors = np.array([0.08177, 0.00017, 0.00723])

fig, ax = plt.subplots(figsize=(4, 4))
x = np.arange(len(metrics))

ax.bar(x, values, yerr=errors, capsize=5, color=['C0', 'C1', 'C2'], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0.9, 1.01)  # zoom into high performance region
ax.set_ylabel('Score')
ax.set_title('Policy Performance with Uncertainty')

plt.tight_layout()
plt.show()


exit()
cm = np.array([[TN, FP],
               [FN, TP]])

# Uncertainties matrix (same shape)
cm_err = np.array([[TN_err, FP_err],
                   [FN_err, TP_err]])

fig, ax = plt.subplots(figsize=(4, 4))

im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Normalized value')

# Labels
class_names = ['Negative', 'Positive']
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(['Pred ' + c for c in class_names])
ax.set_yticklabels(['True ' + c for c in class_names])

# Rotate the x labels if you want
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Annotate each cell with value ± error
fmt = '{:.5f}±{:.5f}'
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        err = cm_err[i, j]
        text_color = 'white' if value > cm.max() / 2. else 'black'
        ax.text(j, i, fmt.format(value, err),
                ha='center', va='center', color=text_color, fontsize=8)

ax.set_title('Normalized Confusion Matrix')
fig.tight_layout()
plt.show()
