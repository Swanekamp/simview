#%%
from p4reader import P4History
import matplotlib.pyplot as plt
import os
'''
Example of loading and plotting a history file using P4History.
This can be used as a standalone script or dropped into a Jupyter notebook 
as a template for more complex analysis.
'''
os.chdir("/home/swane/Runs/Xcimer/KJC/2D/run1")  # Replace with actual path

# Load file
hist = P4History("history.p4")

# Print summary
print(hist.summary())

# --- NEW: list available short labels ---
print("\nAvailable short labels:")
for j, short in enumerate(hist.short_labels):
    unit = hist.units[j]
    print(f"  {short:30s} ({unit})")

# Example: pick one
key = hist.short_labels[1]   # or manually choose
#%%
# Access time and a trace
key = "vin_right"  # example short label, replace with one from the list above
t = hist.time
I = hist[key]   # using short label

# Plot
plt.figure()
plt.plot(t, I)
plt.xlabel("Time (ns)")
plt.ylabel(hist.get_unit(key))
plt.title(hist.get_label(key))
plt.grid(True)
plt.show()
# %%
