
t = []
def get_parameters(obj):
    for i in obj.__dict__.values():
        if i is not None:
            t.append(i)

    return t
