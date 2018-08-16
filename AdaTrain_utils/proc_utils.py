def normalize(v):
    for i in range(len(v)):
        v[i] = v[i] / v[i].max()
    return v
