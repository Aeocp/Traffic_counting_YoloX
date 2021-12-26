def LineCrossing(midpoint,p_midpoint,line0,line1):
  A = midpoint
  B = p_midpoint
  C = line0
  D = line1
  ccw1 = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (D[0] - A[0])
  ccw2 = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
  ccw3 = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
  ccw4 = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (D[0] - A[0])
  if (ccw1 != ccw2 and ccw3 != ccw4):
    return True
  else:
    return False
