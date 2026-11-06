import os

# Workaround for the "OMP: Error #15 ... libiomp5md.dll already initialized" crash
# that appears on some Windows/conda installs where PyTorch and NumPy ship
# different OpenMP runtimes. setdefault() leaves an explicit user setting untouched.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
