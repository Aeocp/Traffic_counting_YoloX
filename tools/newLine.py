def createLine():
  numLine = int(input("Enter number of counting line :"))
  x = []
  y = []
  print("Enter the beginning and ending of the line (0-1) (xb,yb,xe,ye) :")
  for l in range(numLine):
    print("Line",l+1)
    Bp = input("Enter line position :").split(",")
    x.append((Bp[0]))
    y.append((Bp[1]))
    x.append((Bp[2]))
    y.append((Bp[3]))
  return numLine,x,y
