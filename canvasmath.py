import numpy as np
from numpy.linalg import norm


class CanvasMath:
    class LinAlgError(Exception):
        """Base class for exceptions in the CanvasMath library."""

        def __init__(self, message=None):
            super().__init__(message)

    class Vector4:
        def __init__(self, data):
            if isinstance(data, CanvasMath.Vector4):
                self.vector = data.vector.copy()
            elif isinstance(data, list):
                if len(data) == 3:
                    self.vector = np.array(data + [1], dtype=float)
                elif len(data) == 4:
                    self.vector = np.array(data, dtype=float)
                else:
                    raise ValueError(
                        "Vector4 list data must have length 3 (x, y, z) or 4 (x, y, z, w)."
                    )
            elif isinstance(data, np.ndarray):
                if not np.issubdtype(data.dtype, np.number):
                    raise TypeError("Vector4 array must contain numeric values.")
                if data.size == 3:
                    self.vector = np.array([data[0], data[1], data[2], 1], dtype=float)
                elif data.size == 4:
                    self.vector = data.astype(float)
                else:
                    raise ValueError("Vector4 array must have size 3 or 4.")
            else:
                raise TypeError(
                    f"Invalid input type for Vector4: {type(data)}. Must be a list or NumPy array."
                )

        def __repr__(self):
            return f"LinAlg.Vector4({self.vector[0]}, {self.vector[1]}, {self.vector[2]}, {self.vector[3]})"

        def __add__(self, other):
            if isinstance(other, CanvasMath.Vector4):
                return CanvasMath.Vector4(self.vector + other.vector)
            elif isinstance(other, np.ndarray) and other.size == 4:
                return CanvasMath.Vector4(self.vector + other.astype(float))
            else:
                raise TypeError(f"Cannot add Vector4 with type: {type(other)}")

        def __sub__(self, other):
            if isinstance(other, CanvasMath.Vector4):
                return CanvasMath.Vector4(self.vector - other.vector)
            elif isinstance(other, np.ndarray) and other.size == 4:
                return CanvasMath.Vector4(self.vector - other.astype(float))
            else:
                raise TypeError(f"Cannot subtract Vector4 with type: {type(other)}")

        def __mul__(self, scalar):
            if isinstance(scalar, (int, float, np.number)):
                return CanvasMath.Vector4(self.vector * scalar)
            else:
                raise TypeError(f"Cannot multiply Vector4 with type: {type(scalar)}")

        def __rmul__(self, scalar):
            return self.__mul__(scalar)

        def dot(self, other):
            if isinstance(other, CanvasMath.Vector4):
                return np.dot(self.vector, other.vector)
            elif isinstance(other, np.ndarray) and other.size == 4:
                return np.dot(self.vector, other.astype(float))
            else:
                raise TypeError(
                    f"Cannot calculate dot product with type: {type(other)}"
                )

        def to_numpy(self):
            return self.vector

        def to_vector3(self):
            return CanvasMath.Vector(self.vector[:3])

    class Vector:
        def __init__(self, data):
            if isinstance(data, CanvasMath.Vector):
                self.vector = data.vector.copy()
            elif isinstance(data, list):
                if len(data) == 3:
                    self.vector = np.array(data + [0], dtype=float)
                elif len(data) == 4:
                    self.vector = np.array(data, dtype=float)
                else:
                    raise ValueError(
                        "Vector list data must have length 3 (x, y, z) or 4 (x, y, z, w)."
                    )
            elif isinstance(data, np.ndarray):
                if not np.issubdtype(data.dtype, np.number):
                    raise TypeError("Vector array must contain numeric values.")
                if data.size == 3:
                    self.vector = np.array([data[0], data[1], data[2], 0], dtype=float)
                elif data.size == 4:
                    self.vector = data.astype(float)
                else:
                    raise ValueError("Vector array must have size 3 or 4.")
            else:
                raise TypeError(
                    f"Invalid input type for Vector: {type(data)}. Must be a list or NumPy array."
                )

        def __add__(self, other):
            if isinstance(other, CanvasMath.Vector):
                return CanvasMath.Vector(self.vector + other.vector)
            elif isinstance(other, np.ndarray) and other.size == 3:
                return CanvasMath.Vector(self.vector + other.astype(float))
            else:
                raise TypeError(f"Cannot add Vector with type: {type(other)}")

        def __sub__(self, other):
            if isinstance(other, CanvasMath.Vector):
                return CanvasMath.Vector(self.vector - other.vector)
            elif isinstance(other, np.ndarray) and other.size == 3:
                return CanvasMath.Vector(self.vector - other.astype(float))
            else:
                raise TypeError(f"Cannot subtract Vector with type: {type(other)}")

        def __mul__(self, scalar):
            if isinstance(scalar, (int, float, np.number)):
                return CanvasMath.Vector(self.vector * scalar)
            else:
                raise TypeError(f"Cannot multiply Vector with type: {type(scalar)}")

        def __rmul__(self, scalar):
            return self.__mul__(scalar)

        def dot(self, other):
            if isinstance(other, CanvasMath.Vector):
                return np.dot(self.vector, other.vector)
            elif isinstance(other, np.ndarray) and other.size == 3:
                return np.dot(self.vector, other.astype(float))
            else:
                raise TypeError(
                    f"Cannot calculate dot product with type: {type(other)}"
                )

        def cross(self, other):
            if isinstance(other, CanvasMath.Vector):
                return CanvasMath.Vector(np.cross(self.vector, other.vector))
            elif isinstance(other, np.ndarray) and other.size == 3:
                return CanvasMath.Vector(np.cross(self.vector, other.astype(float)))
            else:
                raise TypeError(
                    f"Cannot calculate cross product with type: {type(other)}"
                )

        def magnitude(self):
            return np.linalg.norm(self.vector)

        def normalize(self):
            return CanvasMath.Vector(self.vector / np.linalg.norm(self.vector))

        def to_numpy(self):
            return self.vector

    class Matrix:
        def __init__(self, data):
            if isinstance(data, np.ndarray):
                if not np.issubdtype(data.dtype, np.number):
                    raise TypeError("Matrix data must be numeric.")
                if data.shape == (4, 4):
                    self.matrix = data.astype(float)
                else:
                    raise ValueError("Matrix data must be a 4x4 array.")
            elif isinstance(data, CanvasMath.Matrix):
                self.matrix = data.matrix.copy()
            elif isinstance(data, list):
                if len(data) != 4 or not all(
                    isinstance(row, list) and len(row) == 4 for row in data
                ):
                    raise ValueError(
                        "Matrix list data must be a list of 4 lists, each containing 4 numbers."
                    )
                self.matrix = np.array(data, dtype=float)
            else:
                raise TypeError(
                    "Matrix constructor expects a 4x4 NumPy array, a LinAlg.Matrix object, or a list of lists."
                )

        # Arithmetic operators (static)
        @staticmethod
        def add(matrix1, matrix2):
            if not isinstance(matrix1, CanvasMath.Matrix) or not isinstance(
                matrix2, CanvasMath.Matrix
            ):
                raise TypeError("Inputs must be LinAlg.Matrix objects.")
            return CanvasMath.Matrix(matrix1.matrix + matrix2.matrix)

        @staticmethod
        def subtract(matrix1, matrix2):
            if not isinstance(matrix1, CanvasMath.Matrix) or not isinstance(
                matrix2, CanvasMath.Matrix
            ):
                raise TypeError("Inputs must be LinAlg.Matrix objects.")
            return CanvasMath.Matrix(matrix1.matrix - matrix2.matrix)

        @staticmethod
        def multiply(matrix1, other):
            if not isinstance(matrix1, CanvasMath.Matrix):
                raise TypeError("First input must be a LinAlg.Matrix object.")
            if isinstance(other, CanvasMath.Matrix):
                return CanvasMath.Matrix(np.dot(matrix1.matrix, other.matrix))
            elif isinstance(other, CanvasMath.Vector4):
                return CanvasMath.Vector4(np.dot(matrix1.matrix, other.vector))
            elif isinstance(other, (int, float, np.number)):
                return CanvasMath.Matrix(matrix1.matrix * other)
            else:
                raise TypeError(f"Cannot multiply Matrix with type: {type(other)}")

        # Other matrix operations (static)
        @staticmethod
        def transpose(matrix):
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")
            return CanvasMath.Matrix(matrix.matrix.transpose())

        @staticmethod
        def determinant(matrix):
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")
            return np.linalg.det(matrix.matrix)

        @staticmethod
        def inverse(matrix):
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")
            return CanvasMath.Matrix(np.linalg.inv(matrix.matrix))

        # Advanced linear algebra operations (static)
        @staticmethod
        def lu_decomposition(matrix):
            """Performs LU decomposition on the matrix (in-place)."""
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")

            n = matrix.matrix.shape[0]  # Get the size of the matrix
            L = np.zeros_like(matrix.matrix)  # Initialize L and U
            U = np.zeros_like(matrix.matrix)
            P = np.identity(n)  # Initialize the permutation matrix

            for i in range(n):
                # Find the pivot element
                max_row = i
                for j in range(i + 1, n):
                    if abs(matrix.matrix[j, i]) > abs(matrix.matrix[max_row, i]):
                        max_row = j

                # Swap rows if necessary (for pivoting)
                if max_row != i:
                    matrix.matrix[[i, max_row]] = matrix.matrix[[max_row, i]]
                    P[[i, max_row]] = P[[max_row, i]]
                    L[[i, max_row]] = L[[max_row, i]]

                # Calculate L and U
                L[i, i] = 1.0
                for j in range(i, n):
                    U[i, j] = matrix.matrix[i, j]
                    for k in range(i):
                        U[i, j] -= L[i, k] * U[k, j]

                for j in range(i + 1, n):
                    L[j, i] = matrix.matrix[j, i]
                    for k in range(i):
                        L[j, i] -= L[j, k] * U[k, i]
                    L[j, i] /= U[i, i]

            return CanvasMath.Matrix(L), CanvasMath.Matrix(U), CanvasMath.Matrix(P)

        @staticmethod
        def qr_decomposition(matrix):
            """Performs QR decomposition on the matrix."""
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")
            Q, R = np.linalg.qr(matrix.matrix)
            return CanvasMath.Matrix(Q), CanvasMath.Matrix(R)

        @staticmethod
        def svd(matrix):
            """Performs Singular Value Decomposition (SVD) on the matrix."""
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")
            U, S, Vh = np.linalg.svd(matrix.matrix)
            S = np.diag(S)  # Convert singular values to a diagonal matrix
            return CanvasMath.Matrix(U), CanvasMath.Matrix(S), CanvasMath.Matrix(Vh)

        @staticmethod
        def cholesky_decomposition(matrix):
            """Performs Cholesky decomposition on the matrix."""
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")
            try:
                L = np.linalg.cholesky(matrix.matrix)
                return CanvasMath.Matrix(L)
            except np.linalg.LinAlgError as e:
                raise CanvasMath.LinAlgError(str(e)) from e

        @staticmethod
        def solve_linear_system(A, b):
            """Solves a system of linear equations Ax = b."""
            if not isinstance(A, CanvasMath.Matrix) or not isinstance(
                b, CanvasMath.Vector4
            ):
                raise TypeError(
                    "Inputs must be LinAlg.Matrix and LinAlg.Vector4 objects."
                )
            # print("A matrix:\n", A.matrix)
            # print("b vector:\n", b.to_numpy())

            x = np.linalg.solve(A.matrix, b.to_numpy())

            # print(f"Solution {x[0], x[1], x[2], x[3]}")

            return CanvasMath.Vector4(x)  # Pass the entire NumPy array 'x'

        @staticmethod
        def eigenvalues_eigenvectors(matrix):
            """Calculates the eigenvalues and eigenvectors of the matrix."""
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")
            eigenvalues, eigenvectors = np.linalg.eig(matrix.matrix)
            # Normalize eigenvectors
            for i in range(eigenvectors.shape[1]):
                eigenvectors[:, i] /= norm(eigenvectors[:, i])
            return CanvasMath.Vector(eigenvalues), CanvasMath.Matrix(eigenvectors)

        @staticmethod
        def matrix_norm(matrix, p=None):
            """Calculates the matrix norm."""
            print(f"Matrix: \n{matrix.matrix}")
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")
            if p == "fro":
                print(f"fro: {np.sqrt(np.sum(np.square(matrix.matrix)))}")
                return np.sqrt(np.sum(np.square(matrix.matrix)))  # Frobenius norm
            else:
                print(f"norm: {norm(matrix.matrix, p)}")
                return norm(matrix.matrix, p)  # Use NumPy's norm for other norms

        @staticmethod
        def trace(matrix):
            """Calculates the trace of the matrix."""
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")
            return np.trace(matrix.matrix)

        @staticmethod
        def rank(matrix):
            """Calculates the rank of the matrix."""
            if not isinstance(matrix, CanvasMath.Matrix):
                raise TypeError("Input must be a LinAlg.Matrix object.")
            return np.linalg.matrix_rank(matrix.matrix)

        # Transformation matrices (static)
        @staticmethod
        def translation(x, y, z):
            """Creates a translation matrix."""
            if not all(isinstance(arg, (int, float, np.number)) for arg in (x, y, z)):
                raise TypeError("Translation values must be numeric.")
            matrix = np.identity(4, dtype=float)
            matrix[0, 3] = x
            matrix[1, 3] = y
            matrix[2, 3] = z
            return CanvasMath.Matrix(matrix)

        @staticmethod
        def scaling(x, y, z):
            """Creates a scaling matrix."""
            if not all(isinstance(arg, (int, float, np.number)) for arg in (x, y, z)):
                raise TypeError("Scaling values must be numeric.")
            matrix = np.identity(4, dtype=float)
            matrix[0, 0] = x
            matrix[1, 1] = y
            matrix[2, 2] = z
            return CanvasMath.Matrix(matrix)

        @staticmethod
        def rotation_x(angle_radians):
            """Creates a rotation matrix around the x-axis."""
            if not isinstance(angle_radians, (int, float, np.number)):
                raise TypeError("Rotation angle must be numeric.")
            c = np.cos(angle_radians)
            s = np.sin(angle_radians)
            return CanvasMath.Matrix(
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, c, -s, 0],
                        [0, s, c, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=float,
                )
            )

        @staticmethod
        def rotation_y(angle_radians):
            """Creates a rotation matrix around the y-axis."""
            if not isinstance(angle_radians, (int, float, np.number)):
                raise TypeError("Rotation angle must be numeric.")
            c = np.cos(angle_radians)
            s = np.sin(angle_radians)
            return CanvasMath.Matrix(
                np.array(
                    [
                        [c, 0, s, 0],
                        [0, 1, 0, 0],
                        [-s, 0, c, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=float,
                )
            )

        @staticmethod
        def rotation_z(angle_radians):
            """Creates a rotation matrix around the z-axis."""
            if not isinstance(angle_radians, (int, float, np.number)):
                raise TypeError("Rotation angle must be numeric.")
            c = np.cos(angle_radians)
            s = np.sin(angle_radians)
            return CanvasMath.Matrix(
                np.array(
                    [
                        [c, -s, 0, 0],
                        [s, c, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=float,
                )
            )

        @staticmethod
        def perspective_projection(fov_y_radians, aspect_ratio, near, far):
            """Creates a perspective projection matrix."""
            if not all(
                isinstance(arg, (int, float, np.number))
                for arg in (fov_y_radians, aspect_ratio, near, far)
            ):
                raise TypeError("Perspective projection parameters must be numeric.")
            f = 1.0 / np.tan(fov_y_radians / 2.0)
            return CanvasMath.Matrix(
                np.array(
                    [
                        [f / aspect_ratio, 0, 0, 0],
                        [0, f, 0, 0],
                        [
                            0,
                            0,
                            -(far + near) / (far - near),
                            -(2 * far * near) / (far - near),
                        ],
                        [0, 0, -1, 0],
                    ],
                    dtype=float,
                )
            )

        @staticmethod
        def orthographic_projection(left, right, bottom, top, near, far):
            """Creates an orthographic projection matrix."""
            if not all(
                isinstance(arg, (int, float, np.number))
                for arg in (left, right, bottom, top, near, far)
            ):
                raise TypeError("Orthographic projection parameters must be numeric.")
            return CanvasMath.Matrix(
                np.array(
                    [
                        [2 / (right - left), 0, 0, -(right + left) / (right - left)],
                        [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                        [0, 0, 0, 1],
                    ],
                    dtype=float,
                )
            )

        # Instance methods for applying transformations
        def apply_translation(self, x, y, z):
            """Applies translation to the matrix in-place."""
            self.matrix = CanvasMath.Matrix.multiply(
                CanvasMath.Matrix.translation(x, y, z), self
            ).matrix

        def apply_scaling(self, x, y, z):
            """Applies scaling to the matrix in-place."""
            self.matrix = CanvasMath.Matrix.multiply(
                CanvasMath.Matrix.scaling(x, y, z), self
            ).matrix

        def apply_rotation_x(self, angle_radians):
            """Applies rotation around the x-axis in-place."""
            self.matrix = CanvasMath.Matrix.multiply(
                CanvasMath.Matrix.rotation_x(angle_radians), self
            ).matrix

        def apply_rotation_y(self, angle_radians):
            """Applies rotation around the y-axis in-place."""
            self.matrix = CanvasMath.Matrix.multiply(
                CanvasMath.Matrix.rotation_y(angle_radians), self
            ).matrix

        def apply_rotation_z(self, angle_radians):
            """Applies rotation around the z-axis in-place."""
            self.matrix = CanvasMath.Matrix.multiply(
                CanvasMath.Matrix.rotation_z(angle_radians), self
            ).matrix

        # ... (Other apply_* methods for perspective and orthographic projections)

        def to_numpy(self):
            return self.matrix

        def __repr__(self):
            return f"LinAlg.Matrix:\n{self.matrix}"

        def __neg__(self):
            """Returns a new Matrix with negated elements."""
            return CanvasMath.Matrix(-self.matrix)

        def __getitem__(self, index):
            """Allows accessing matrix elements using indexing."""
            return self.matrix[index]

        def __setitem__(self, index, value):
            """Allows modifying matrix elements using indexing."""
            self.matrix[index] = value
