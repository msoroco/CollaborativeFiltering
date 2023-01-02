

def apply(source, dest, indices):
    a = source.flatten()
    a[indices] = dest.flatten()[indices]
    a = a.reshape(dest.shape)
    return a