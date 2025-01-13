# 🚀 **Parallel Computing and Architecture (PCAP) - 6th Semester**

Welcome to my **PCAP (Parallel Computing and Architecture)** repository! This repo showcases the **lab exercises** and **projects** completed during my **6th Semester** at **MIT Manipal**, focusing on **parallel computing** and **MPI (Message Passing Interface)**.

---

## 🛠️ **Technologies Used**

In this course, I have worked with the following technologies:

- **C Programming**: The primary language for implementation.
- **MPI (Message Passing Interface)**: Used for parallel programming and inter-process communication.
- **Makefiles**: To streamline the build process and make project management easier.

---

## 🎯 **Purpose of this Repository**

This repository contains solutions to **PCAP** lab exercises and projects, focusing on:

- 📚 **MPI-based parallel exercises**  
- 💻 **Parallel computing & distributed systems**  
- ⚡ **Exploring parallel algorithms**  

---

## 💡 **Learning Outcomes**

Throughout this project, I have gained hands-on experience in the following areas:

- 🧠 **Parallel Algorithms**: Developing and implementing parallel algorithms using MPI.
- 🌐 **Distributed Systems**: Solving complex problems in distributed environments.
- 🔧 **C Programming**: Enhancing memory management and coding efficiency in C.

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

- 🧮 **Power Calculation**: using MPI across multiple processes.
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

🚀 **Happy Learning and Coding!**

---
