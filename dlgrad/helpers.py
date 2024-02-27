class ShapeError(Exception): ...

def get_list_dim(data: list, dim=0):
    if isinstance(data, list): 
        dim +=1
        return get_list_dim(data[0], dim)
    else: return dim