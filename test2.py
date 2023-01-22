

class a:
  bg = False
  def __init__(self) -> None:
    print("in init")

  def pp(self):
    print("in pp")
    print(a.bg)

# a().pp()