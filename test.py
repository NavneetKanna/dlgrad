# from dlgrad import Tensor

def foo(*args, **kwargs):
    print(*args, args)
    # print(**kwargs)
    print(kwargs)


foo(x=2, y=0)
