def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	row_a = len(a)
	col_a = len(a[0])
	row_b = len(b)

	if col_a != row_b:
		return -1
	else:
		ans = []
		for i in range(row_a):
			sum = 0
			for j in range(col_a):
					sum = sum + (a[i][j]*b[j])
			ans.append(sum)
		return ans

	pass