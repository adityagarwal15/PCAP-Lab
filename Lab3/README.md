# 🔥 **Lab 3: Collective Communications in MPI**

Welcome to **Lab 3** of the **Parallel Computing and Architecture (PCAP)** course! 🚀  
In this lab, I focused on **collective communication** in **MPI (Message Passing Interface)**. The lab involved creating parallel algorithms using collective operations like `MPI_Scatter`, `MPI_Gather`, and `MPI_Reduce` to facilitate communication across multiple processes.

---

## 🧮 **Lab Tasks**

Here are the key tasks I tackled in this lab:

1. **Distributed Factorial Calculation**: Implemented a program where the root process reads an integer `N` and distributes values among processes. Each process calculates the factorial of its assigned value, and the root process gathers the results and computes their sum using `MPI_Scatter` and `MPI_Reduce`.
2. **Array Average Computation**: Designed a program where the root process reads a 1D array of size `N × M`, distributes `M` elements to each process, and each process computes the average of its elements. The root process collects the averages using `MPI_Scatter` and `MPI_Gather` and calculates the total average.
3. **Counting Non-Vowels**: Created a program where the root process reads a string and divides it equally among processes. Each process counts the non-vowel characters in its substring and sends the count to the root. The root displays the counts from all processes and the total count using `MPI_Scatter` and `MPI_Gather`.
4. **String Interleaving**: Developed a program where the root process reads two strings of equal length, divides the strings into segments, and distributes them to processes. Each process interleaves characters from the corresponding substrings. The root gathers the interleaved segments and displays the final result using `MPI_Scatter`, `MPI_Gather`, and `MPI_Bcast`.

---

## 🚀 **Learning Objectives**

By completing this lab, I enhanced my skills in:

- **Collective Communication**: Mastering the use of `MPI_Scatter`, `MPI_Gather`, `MPI_Reduce`, and `MPI_Bcast` to facilitate communication between multiple processes.
- **Parallel Programming**: Writing parallel programs where processes work together to compute results using collective operations.
- **Efficient Data Handling**: Understanding how to distribute, collect, and aggregate data efficiently across processes.
- **Task Distribution**: Managing the partitioning of tasks and data among processes to ensure proper load balancing.

---

## 📂 **Code Structure**

The following files contain the implementations of the tasks:

- **`factorial.c`**: MPI program for distributed factorial calculation using `MPI_Scatter` and `MPI_Reduce`.
- **`average.c`**: Code implementing array average computation where `M` elements are distributed to each process, and averages are gathered.
- **`vowels.c`**: Program for counting non-vowels in a string, where the string is divided among processes.
- **`interleave.c`**: String interleaving program where processes interleave two strings and the root displays the result.

---

## 🔗 **Explore Other Labs**

For more exercises and labs, check out the [main repository](https://github.com/adityagarwal15/PCAP-Lab).

---

🚀 **Happy Learning and Coding!** 💻✨
