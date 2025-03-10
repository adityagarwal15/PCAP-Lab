# 🚀 **Lab 8: Matrix Operations in CUDA**

Welcome to **Lab 8** of the **Parallel Computing and Architecture (PCAP)** course! 🌟  
This lab focuses on **parallel matrix operations** using CUDA programming. You'll implement addition and multiplication of matrices using different parallel computing approaches.

---
## 🧑‍💻 **Lab Exercises**

### 1. **Matrix Addition in CUDA**
Implement a CUDA program that adds two matrices using three different parallel strategies:

#### **Specifications:**
A. Each row of the resultant matrix is computed by one thread.  
B. Each column of the resultant matrix is computed by one thread.  
C. Each element of the resultant matrix is computed by one thread.  

#### **Key Features:**
- Uses different CUDA grid and block configurations.
- Efficient memory management with proper CUDA allocations.
- Takes input from the user dynamically.

#### **Example Usage:**
```bash
Enter number of rows: 3
Enter number of columns: 3
Enter elements of Matrix A: 
1 2 3
4 5 6
7 8 9
Enter elements of Matrix B: 
9 8 7
6 5 4
3 2 1
Resultant Matrix C:
10 10 10
10 10 10
10 10 10
```

#### **Implementation Details:**
- Three different CUDA kernels handle row-wise, column-wise, and element-wise addition.
- Each thread computes only its assigned portion of the matrix.
- The program ensures correct memory access and synchronization.

---
### 2. **Matrix Multiplication in CUDA**
Implement a CUDA program that multiplies two matrices using three different parallel strategies:

#### **Specifications:**
A. Each row of the resultant matrix is computed by one thread.  
B. Each column of the resultant matrix is computed by one thread.  
C. Each element of the resultant matrix is computed by one thread.  

#### **Key Features:**
- Uses efficient parallel computation techniques.
- Handles dynamic user input for matrices.
- Supports different grid and block configurations for optimization.

#### **Example Usage:**
```bash
Enter number of rows for Matrix A: 2 2
Enter number of columns for Matrix A (rows for B): 2 2
Enter number of columns for Matrix B: 2
Enter elements of Matrix A: 
1 2 3
4 5 6
Enter elements of Matrix B: 
7 8
9 10
11 12
Resultant Matrix C:
58 64
139 154
```

#### **Implementation Details:**
- Uses three CUDA kernels to perform row-wise, column-wise, and element-wise multiplication.
- Optimized memory access for performance.
- Supports various matrix sizes with proper validation.

---
## 🎯 **Learning Objectives**
By completing this lab, you will:
- **Understand Parallel Matrix Operations**: Implement addition and multiplication using different parallel strategies.
- **Optimize CUDA Kernels**: Use efficient thread configurations for computation.
- **Manage Memory in CUDA**: Allocate, transfer, and deallocate GPU memory properly.
- **Handle User Input in CUDA Programs**: Accept matrix dimensions and values dynamically.

---
## 📂 **Code Structure**
The lab contains two main CUDA programs:
- **`matrix_addition.cu`**: Implements matrix addition using three different approaches.
- **`matrix_multiplication.cu`**: Implements matrix multiplication using three different approaches.

---
## 📚 **Compilation Instructions**

### Matrix Addition Program:
```bash
nvcc matrix_addition.cu -o matrix_addition
./matrix_addition
```

### Matrix Multiplication Program:
```bash
nvcc matrix_multiplication.cu -o matrix_multiplication
./matrix_multiplication
```

---
## 🤝 **Contributing**
Feel free to suggest improvements or report issues through:
- Lab feedback sessions.
- Course forums.
- Direct communication with instructors.

---
🚀 **Happy Coding and Learning!** 💻✨
