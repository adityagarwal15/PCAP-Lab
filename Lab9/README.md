# 🚀 **Lab 9: Advanced Matrix Operations in CUDA**

Welcome to **Lab 9** of the **Parallel Computing and Architecture (PCAP)** course! 🌟  
This lab focuses on **advanced parallel matrix operations** using CUDA programming, including sparse matrix-vector multiplication and specialized matrix transformations.

---
## 🧑‍💻 **Lab Exercises**

### 1. **Sparse Matrix-Vector Multiplication using CSR Format**
Implement a CUDA program that performs parallel Sparse Matrix-Vector multiplication using the Compressed Sparse Row (CSR) storage format.

#### **Specifications:**
- Represent the input sparse matrix in CSR format in the host code
- Efficiently transfer the CSR data structures to the GPU
- Implement a parallel SpMV kernel that properly handles the sparse format

#### **Key Features:**
- Efficient memory usage through the CSR format, storing only non-zero elements
- Parallel computation across rows of the sparse matrix
- Optimized thread workload distribution

#### **Implementation Details:**
- CSR format consists of three arrays:
  - `values`: Stores all non-zero values
  - `col_indices`: Stores column indices of each non-zero value
  - `row_ptrs`: Stores indices into the values array for the start of each row
- Each thread computes the result for one row of the sparse matrix

---
### 2. **Matrix Element Transformation by Row Position**

Implement a CUDA program to read an M×N matrix A and transform its elements based on their row position:
- 1st row: Keep elements unchanged
- 2nd row: Replace each element with its square
- 3rd row: Replace each element with its cube
- And so on...

#### **Specifications:**
- Create a kernel that applies different power operations based on row index
- Use efficient parallel computation techniques
- Support matrices of different dimensions

#### **Key Features:**
- Row-based transformations using thread indices
- Dynamic exponent calculation based on row number
- Efficient memory access patterns

#### **Example:**
```
Input Matrix A:
2 3 1
4 5 6
2 1 3

Output Matrix:
2  3  1   (unchanged)
16 25 36  (squared)
8  1  27  (cubed)
```

---
### 3. **Border Preservation with 1's Complement Transformation**

Implement a CUDA program that reads a matrix A of size M×N and produces an output matrix B of the same size where:
- Border elements remain unchanged
- Non-border elements are replaced with their 1's complement

#### **Specifications:**
- Identify border vs. non-border elements using thread indices
- Apply 1's complement operation to non-border elements only
- Preserve original values of border elements

#### **Key Features:**
- Conditional processing based on element position
- Efficient parallel implementation with one thread per matrix element
- Boundary checking to identify border elements

#### **Example:**
```
Input Matrix A:
1 2 3 4
6 5 8 3
2 4 10 1
9 1 2 5

Output Matrix B:
1 2  3  4
6 10 7  3
2 11 5  1
9 1  2  5
```

---
## 🎯 **Learning Objectives**
By completing this lab, you will:
- **Master Sparse Matrix Operations**: Implement efficient sparse matrix algorithms using the CSR format
- **Apply Conditional Transformations**: Develop kernels that perform different operations based on element position
- **Optimize Memory Access Patterns**: Efficiently handle different matrix storage formats
- **Implement Position-Based Logic**: Apply different transformations based on element coordinates in the matrix

---
## 📂 **Code Structure**
The lab contains three main CUDA programs:
- **sparse.cu**: Implements sparse matrix-vector multiplication using CSR format
- **transformation.cu**: Implements row-based power transformations
- **complement.cu**: Implements border preservation with 1's complement transformation

---
## 📚 **Compilation Instructions**

### Sparse Matrix-Vector Multiplication:
```bash
nvcc sparse.cu -o sparse
./sparse
```

### Row Power Transformation:
```bash
nvcc transformation.cu -o transformation
./transformation
```

### Border Complement Transformation:
```bash
nvcc complement.cu -o complement
./complement
```

---
## 🤝 **Contributing**
Feel free to suggest improvements or report issues through:
- Lab feedback sessions
- Course forums
- Direct communication with instructors

---
🚀 **Happy Coding and Learning!** 💻✨
