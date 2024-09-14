
t = []
def get_parameters(obj):
    for i in obj.__dict__:
        for j in obj.__dict__[i].__dict__.values():
            if j is not None:
                t.append(j)

    return t
