# 🚀 **Parallel Computing and Architecture (PCAP) - 6th Semester**

Welcome to my **PCAP (Parallel Computing and Architecture)** repository! This repo showcases the **lab exercises** and **projects** completed during my **6th Semester** at **MIT Manipal**, focusing on **parallel computing** and **MPI (Message Passing Interface)**.

---

## 🛠️ **Technologies Used**

In this course, I have worked with the following technologies:

- **C Programming**: The primary language for implementation.
- **MPI (Message Passing Interface)**: Used for parallel programming and inter-process communication.
- **CUDA**: For parallel programming on GPUs.
- **Makefiles**: To streamline the build process and make project management easier.

---

## 🎯 **Purpose of this Repository**

This repository contains solutions to **PCAP** lab exercises and projects, focusing on:

- 📚 **MPI-based parallel exercises**  
- 💻 **Parallel computing & distributed systems**  
- ⚡ **Exploring parallel algorithms**  
- 🔢 **CUDA programming for parallel processing on GPUs**

---

## 💡 **Learning Outcomes**

Throughout this project, I have gained hands-on experience in the following areas:

- 🧠 **Parallel Algorithms**: Developing and implementing parallel algorithms using MPI.
- 🌐 **Distributed Systems**: Solving complex problems in distributed environments.
- 🔧 **C Programming**: Enhancing memory management and coding efficiency in C.
- 🚀 **GPU Programming**: Implementing efficient parallel algorithms using CUDA.

---

## 🔗 **Connect with Me**

Let’s stay connected! You can explore my other projects and reach out via:

- 🌟 [GitHub Profile](https://github.com/adityagarwal15)
- 💼 [LinkedIn Profile](https://www.linkedin.com/in/aditya-agarwal-12601b27b/)
- 🌐 [Portfolio](https://adityagarwal.netlify.app)

---

## 🔥 **Lab Highlights**

### **Lab 1: Introduction to MPI & Parallel Computing**

In **Lab 1**, I delved into the basics of **MPI** and parallel computing. Key tasks included:

- 🧮 **Power Calculation**: Using MPI across multiple processes.
- 🔢 **Factorial & Fibonacci Calculations**: Using different processes for calculation.
- 📝 **Even/Odd Processes**: Printing "Hello" for even-ranked processes and "World" for odd-ranked ones.
- 🔢 **Simple Calculator**: Performing parallel operations (add, subtract, multiply, divide).
- 🔠 **Toggling Characters**: Manipulating strings based on the rank of the process (e.g., "HELLO" → "hElLo").

### 📂 [Explore Lab 1 in Detail](./Lab1)

---

### **Lab 2: Point-to-Point Communications in MPI**

In **Lab 2**, I explored point-to-point communication using MPI. Key tasks included:

- 🔠 **Synchronous Send**: The sender process sends a word to the receiver, and the receiver toggles the case of each letter before sending it back to the sender.
- 🔢 **Master-Slave Communication**: The master process sends a number to each slave process, and each slave receives the number and prints it using standard send operations.
- 🧮 **Squaring and Cubing Array Elements**: The root process sends values to slaves, where even-ranked processes square the number and odd-ranked processes cube it, using buffered send.
- 🔄 **Chain Communication**: The root process sends an integer value to the first process, each subsequent process increments the value by one, and the last process sends the value back to the root using point-to-point communication.

### 📂 [Explore Lab 2 in Detail](./Lab2)

---

### **Lab 3: Collective Communications in MPI**

In **Lab 3**, I explored collective communication techniques using MPI. Key tasks included:

- 🔢 **Distributed Factorial Calculation**:  
  The root process reads an integer `N` and distributes values among processes. Each process calculates the factorial of its assigned value, and the root process gathers all the results to compute their sum using collective routines like `MPI_Scatter` and `MPI_Reduce`.

- 📊 **Array Average Computation**:  
  The root process reads a 1D array of size `N × M` and distributes `M` elements to each process. Each process computes the average of its elements and sends it back to the root, which computes the total average using `MPI_Scatter` and `MPI_Gather`.

- 🔍 **Counting Non-Vowels**:  
  The root process reads a string and divides it equally among processes. Each process counts the non-vowel characters in its substring and sends the count to the root. The root process displays the counts from all processes and the total count using `MPI_Scatter` and `MPI_Gather`.

- 🔀 **String Interleaving**:  
  The root process reads two strings of equal length, divides the strings into segments, and distributes the segments to all processes. Each process interleaves characters from the corresponding substrings, and the root gathers the interleaved segments to form and display the final result using `MPI_Scatter`, `MPI_Gather`, and `MPI_Bcast`.

### 📂 [Explore Lab 3 in Detail](./Lab3)

---

### **Lab 4: Collective Communications and Error Handling in MPI**

In **Lab 4**, I focused on error handling in MPI. Key tasks included:

- 🧠 **Error Detection**: Handling errors during MPI communication and reporting them.
- 🔧 **Collective Operations**: Working with more advanced collective communication operations like `MPI_Bcast` and `MPI_Gather`.
- 📝 **Synchronization Issues**: Managing synchronization issues that arise in distributed systems.

### 📂 [Explore Lab 4 in Detail](./Lab4)

---

### **Lab 5: Programs on Arrays in CUDA**

In **Lab 5**, I began exploring parallel programming with CUDA. Key tasks included:

- 🔢 **Array Initialization**: Implementing array manipulation programs using CUDA for parallel processing.
- 🧮 **Element-wise Operations**: Performing basic array operations (addition, multiplication) in parallel using CUDA.
- 🚀 **Optimizing with CUDA**: Making the code more efficient by utilizing the parallelism available in GPU processing.

### 📂 [Explore Lab 5 in Detail](./Lab5)

---

### **Lab 6: Programs on Arrays in CUDA (Continued)**

In **Lab 6**, I continued working on array manipulation using CUDA. Key tasks included:

- 🔢 **Parallel Summation**: Implementing parallel summation of array elements using CUDA.
- 🚀 **Reduction Algorithms**: Using reduction algorithms to optimize parallel summation of large arrays.

### 📂 [Explore Lab 6 in Detail](./Lab6)

---

### **Lab 7: Programs on Strings in CUDA**

In **Lab 7**, I implemented CUDA programs for string manipulation. Key tasks included:

- 🔠 **String Operations**: Performing string operations (e.g., reversing, concatenating) in parallel using CUDA.
- 🚀 **Optimizing String Processing**: Using CUDA to speed up string processing tasks like pattern matching and string comparison.

### 📂 [Explore Lab 7 in Detail](./Lab7)

---

### **Lab 8: Programs on Matrix in CUDA**

In **Lab 8**, I worked with matrix operations using CUDA. Key tasks included:

- 🔢 **Matrix Multiplication**: Implementing matrix multiplication in parallel using CUDA.
- 🚀 **Optimizing Matrix Operations**: Utilizing CUDA to optimize matrix operations, reducing computational time.

### 📂 [Explore Lab 8 in Detail](./Lab8)

---

### **Lab 9: Programs on Matrix in CUDA (Continued)**

In **Lab 9**, I continued working with matrix operations in CUDA. Key tasks included:

- 🔢 **Matrix Transposition**: Implementing matrix transposition in parallel using CUDA.
- 🧮 **Optimizing Matrix Algorithms**: Applying optimization techniques to matrix operations for better performance.

### 📂 [Explore Lab 9 in Detail](./Lab9)

---

### **Lab 10: Programs Using Different CUDA Device Memory Types and Synchronization**

In **Lab 10**, I explored different memory types in CUDA and synchronization techniques. Key tasks included:

- 🧠 **Global, Shared, and Constant Memory**: Exploring different types of CUDA memory and their use cases.
- 🔧 **Synchronization**: Implementing synchronization techniques to ensure correct parallel execution.

### 📂 [Explore Lab 10 in Detail](./Lab10)

---

## 📚 **References**

A list of resources and references used throughout the course.

---

🚀 **Happy Learning and Coding!**
