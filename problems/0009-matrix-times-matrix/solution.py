def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
            #   import numpy as np
            #   c = np.matmul((np.array(a)),(np.array(b))).tolist()
            #   return c
            row_a = len(a)
            col_a = len(a[0])
            row_b = len(b)
            col_b = len(b[0])
            if col_a != row_b:
                return -1
            else:
                mat = []
                for i in range(row_a):
                    row = []
                    for j in range(col_b):
                        ans = 0
                        for k in range(col_a):
                            ans = ans + a[i][k]*b[k][j]
                        row.append(ans)
                    mat.append(row)
                return mat
                        
