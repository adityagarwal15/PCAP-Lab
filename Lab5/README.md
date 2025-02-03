# 🚀 **Lab 5: Parallel Sorting and Vector Operations using CUDA**

Welcome to **Lab 5** of the **Parallel Computing and Architecture (PCAP)** course! 🌐  
In this lab, I explored the implementation of **parallel sorting algorithms** and **vector operations** using **CUDA**. The lab focused on writing CUDA programs for vector additions, sorting algorithms, and transforming data using the GPU.

---

## 🧑‍💻 **Lab Exercises**

### 1. **Write a program in CUDA to add two vectors of length N using**:
   - **a) Block size as N**  
   - **b) N threads**

   Implemented a CUDA program to add two vectors of length `N` using different configurations for thread and block sizes.
   - **Part A**: Block size equals `N`.
   - **Part B**: `N` threads, each processing one element.

   *File: `vector_sum.cu`*

---

### 2. **Implement a CUDA program to add two vectors of length N by keeping the number of threads per block as 256 (constant) and vary the number of blocks to handle N elements.**

   Designed a CUDA program that adds two vectors of length `N` using a constant block size of `256` threads per block, with the number of blocks dynamically adjusted to handle `N` elements.

   *File: `vector_sum2`*

---

### 3. **Write a program in CUDA to process a 1D array containing angles in radians to generate sine of the angles in the output array. Use appropriate function.**

   Created a CUDA program that processes a 1D array of angles in radians and computes their sine values in parallel using the appropriate math function.

   *File: `sine_comp.cu`*

---

## 🧑‍💻 **Additional Exercises**

### 1. **Write a program in CUDA to perform linear algebra function of the form y = ax + y, where x and y are vectors and a is a scalar value.**

   Implemented a CUDA program to perform a linear algebra function `y = ax + y`, where `x` and `y` are vectors, and `a` is a scalar value. The operation is performed element-wise in parallel.

   *File: `vector_add_scalar.cu`*

---

### 2. **Write a program in CUDA to sort every row of a matrix using selection sort.**

   Created a CUDA program to sort each row of a matrix using the **Selection Sort** algorithm in parallel. Each row is processed by a separate block of threads.

   *File: `selection_sort.cu`*

---

### 3. **Write a program in CUDA to perform odd-even transposition sort in parallel.**

   Implemented the **Odd-Even Transposition Sort** algorithm in parallel to sort the elements of a matrix. The sorting steps are performed concurrently with multiple threads working in a staggered manner.

   *File: `transposition_sort.cu`*

---

## 🚀 **Learning Objectives**

By completing this lab, I gained hands-on experience and deepened my understanding of:

- **Parallel Vector Operations**: Learned how to use CUDA to perform vector additions efficiently across different block sizes and thread configurations.
- **Parallel Sorting Algorithms**: Implemented and optimized selection sort and odd-even transposition sort in parallel using CUDA.
- **Mathematical Transformations**: Worked with mathematical functions like sine in parallel using the CUDA device functions.
- **CUDA Memory Management**: Gained a deeper understanding of memory allocation, data transfer, and kernel execution in CUDA programs.
- **Optimized Parallel Programming**: Enhanced my skills in distributing tasks across threads and blocks to efficiently utilize the GPU for data processing and sorting.

---

## 📂 **Code Structure**

The following files contain the implementations of the tasks:

- **`vector_sum.cu`**: Vector addition program using block size as `N`.
- **`vector_sum.cu`**: Vector addition program with `N` threads.
- **`vector_sum2.cu`**: Vector addition program using `256` threads per block.
- **`sine_comp.cu`**: CUDA program for sine transformation of angles in radians.
- **`vector_add_scalar.cu`**: CUDA program to perform the linear algebra operation `y = ax + y`.
- **`selection_sort.cu`**: Program to sort each row of a matrix using Selection Sort.
- **`transposition_sort.cu`**: Program for parallel Odd-Even Transposition Sort.

---

## 🔗 **Explore Other Labs**

For more exercises and labs, check out the [main repository](https://github.com/adityagarwal15/PCAP-Lab).

---

🚀 **Happy Learning and Coding!** 💻✨
