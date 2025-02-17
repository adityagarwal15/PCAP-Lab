# 🚀 **Lab 6: Programs on Arrays in CUDA**

Welcome to **Lab 6** of the **Parallel Computing and Architecture (PCAP)** course! 🌐  
In this lab, you will explore CUDA programming with a focus on **one-dimensional arrays**. The exercises will guide you through parallel operations on arrays, including **convolution**, **selection sort**, and **odd-even transposition sort**.

---

## 🧑‍💻 **Lab Exercises**

### 1. **Write a program in CUDA to perform a convolution operation on a one-dimensional input array.**

   In this exercise, you will implement a CUDA program that performs a convolution operation on an input array `N` of size `width` using a mask array `M` of size `mask_width`. The result will be stored in a one-dimensional array `P` of size `width`.
   
   The convolution operation applies a filter (mask) to the input data. Each element of the result is computed as the weighted sum of the corresponding region of the input array.
   
   *File: `convolution.cu`*

---

### 2. **Write a program in CUDA to perform selection sort in parallel.**

   In this task, you will implement the **Selection Sort** algorithm in parallel using CUDA. Each thread will sort one element, and blocks of threads will cooperate to sort the entire array.
   
   The parallelization of Selection Sort improves its performance by dividing the task of finding the minimum element into multiple threads working on different sections of the array.
   
   *File: `selection_sort.cu`*

---

### 3. **Write a program in CUDA to perform odd-even transposition sort in parallel.**

   The **Odd-Even Transposition Sort** is another sorting algorithm that can be parallelized. In this exercise, you will implement this algorithm in CUDA, where threads operate in a staggered manner to perform a series of swaps until the array is sorted.
   
   Odd-even transposition sort operates on consecutive pairs of elements in alternating "odd" and "even" phases to progressively sort the array.
   
   *File: `transposition.cu`*

---

## 🚀 **Learning Objectives**

By completing this lab, you will:

- **Deepen your understanding of CUDA**: Gain hands-on experience working with 1D arrays in parallel.
- **Understand convolution operations**: Implement a parallel convolution operation to process signals or data.
- **Implement parallel sorting algorithms**: Learn how to adapt classic sorting algorithms like selection sort and odd-even transposition sort for parallel execution on the GPU.
- **Work with CUDA blocks and threads**: Apply your knowledge of CUDA thread organization to solve real-world problems using parallel computing techniques.

---

## 📂 **Code Structure**

The following files contain the implementations for the tasks:

- **`convolution.cu`**: CUDA program for performing convolution on a 1D input array.
- **`selection_sort.cu`**: Parallel implementation of the Selection Sort algorithm in CUDA.
- **`transposition.cu`**: Parallel implementation of Odd-Even Transposition Sort in CUDA.

---

## 🔗 **Explore Other Labs**

For more exercises and labs, check out the [main repository](https://github.com/adityagarwal15/PCAP-Lab).

---

🚀 **Happy Learning and Coding!** 💻✨
