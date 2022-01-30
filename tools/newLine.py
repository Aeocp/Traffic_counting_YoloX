def createLine():
  x = []
  y = []
  print("Enter the beginning and ending of the line (0-1) (xb,yb,xe,ye) :")
  Bp = input("Enter line position :").split(",")
  x.append((Bp[0]))
  y.append((Bp[1]))
  x.append((Bp[2]))
  y.append((Bp[3]))
  return numLine,x,y
