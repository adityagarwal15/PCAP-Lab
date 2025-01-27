# 🔥 **Lab 4: Collective Communications and Error Handling in MPI**

Welcome to **Lab 4** of the **Parallel Computing and Architecture (PCAP)** course! 🚀  
In this lab, I explored the advanced topics of **collective communications** and **error handling** in **MPI (Message Passing Interface)**. The lab focused on implementing MPI programs using collective communication routines such as `MPI_Reduce` and `MPI_Scan` and handling errors gracefully with MPI error-handling functions.

---

## 🧮 **Lab Tasks**

Here are the key tasks I tackled in this lab:

1. **Factorial Sum Calculation using `MPI_Reduce`** (Solved Example):  
   Implemented a program where `N` processes calculate the sum of factorials from 1! to N! using `MPI_Reduce` to collect results at the root process. This is the **solved example** provided in the lab instructions and showcases how to perform reduction operations.  
   *File: `factorial_reduce.c`*
   
2. **Factorial Sum Calculation using `MPI_Scan`**:  
   Designed a program similar to the previous one, but used `MPI_Scan` to perform a scan operation, computing the partial sums of factorials across processes.

3. **Matrix Search**:  
   Wrote a program to read a 3x3 matrix and search for a user-entered element. The search is performed using 3 processes, and the occurrences of the element are computed in parallel.

4. **Matrix Transformation**:  
   Developed a program that reads a 4x4 matrix and performs a transformation where each element of the matrix is multiplied by its row index and column index. This transformation is distributed across four processes.

5. **Pattern Generation**:  
   Created a program that reads a word of length `N` and generates a new word by repeating each character according to its position in the string. This is done using `N` processes, including the root process, and the result is displayed at the root.

---

## 🚀 **Learning Objectives**

By completing this lab, I enhanced my skills in:

- **Collective Communication**: Gained a deep understanding of the use of `MPI_Reduce`, `MPI_Scan`, and other collective operations for communication across multiple processes.
- **Error Handling in MPI**: Learned how to handle errors in MPI programs using error-handling routines like `MPI_Errhandler_set`, `MPI_Error_string`, and `MPI_Error_class`.
- **Parallel Programming**: Strengthened my parallel programming skills by using MPI functions to distribute work and collect results from different processes.
- **Data Distribution and Synchronization**: Mastered how to efficiently distribute tasks across processes and synchronize their results using collective communication functions like `MPI_Barrier` and `MPI_Scan`.

---

## 📂 **Code Structure**

The following files contain the implementations of the tasks:

- **`factorial_reduce.c`**: **Solved example** for calculating the sum of factorials using `MPI_Reduce`.
- **`factorial_scan.c`**: Code implementing factorial sum calculation using `MPI_Scan` and error-handling routines.
- **`matrix_search.c`**: Program for matrix search, where the element is searched across processes and results are gathered.
- **`matrix_transformation.c`**: Matrix transformation program using four processes to compute the desired transformation.
- **`pattern_generation.c`**: Pattern generation program where each character in the word is repeated according to its position using `N` processes.

---

## 🔗 **Explore Other Labs**

For more exercises and labs, check out the [main repository](https://github.com/adityagarwal15/PCAP-Lab).

---

🚀 **Happy Learning and Coding!** 💻✨
