class ShapeError(Exception): ...

def get_list_dim(data: list, dim=0):
    if isinstance(data, list): 
        dim +=1
        return get_list_dim(data[0], dim)
    else: return dim

def calculate_stride(shape: tuple):
    if len(shape) == 1:
        return [1]

    stride = [i for i in range(len(shape))]
    
    if len(shape) == 2:
        stride[0] = shape[-1]
        stride[1] = 1
        return stride
    
    for i in range(len(shape) - 2):
        prod = 1
        for j in range(i+1, len(shape)):
            prod *= shape[j]
        stride[i] = prod
    stride[-2] = shape[-1]
    stride[-1] = 1
    
    return stride

def calculate_offset(n=0, c=0, h=0, w=0, N=0, C=0, H=0, W=0):
    return (n * C*H*W) + (c * H*W) + (h * W) + w 