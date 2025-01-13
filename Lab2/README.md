# 🔥 **Lab 2: Point-to-Point Communications in MPI**

Welcome to **Lab 2** of the **Parallel Computing and Architecture (PCAP)** course! 🚀  
In this lab, I focused on **point-to-point communication** in **MPI (Message Passing Interface)**. The lab involved creating parallel algorithms using MPI's send and receive functions to facilitate direct communication between processes.

---

## 🧮 **Lab Tasks**

Here are the key tasks I tackled in this lab:

1. **Synchronous Send**: Implemented a program where the sender process sends a word to the receiver. The receiver toggles each letter of the word and sends it back to the sender using synchronous send operations.
2. **Master-Slave Communication**: Built a program where the master process (Process 0) sends a number to each slave process, and each slave prints the received number using standard send operations.
3. **Array Manipulation (Squaring & Cubing)**: The root process reads an array of values, and even-ranked processes square the received values while odd-ranked processes cube them. This was implemented using buffered send for efficient communication.
4. **Chain Communication (Incrementing Value)**: Designed a chain communication program where the root process sends an integer to Process 1, and each subsequent process increments the value before passing it along. The last process sends the value back to the root using point-to-point communication.

---

## 🚀 **Learning Objectives**

By completing this lab, I enhanced my skills in:

- **Point-to-Point Communication**: Mastering the use of `MPI_Send`, `MPI_Recv`, and different communication techniques (synchronous send, standard send, and buffered send).
- **Parallel Programming**: Writing parallel programs that allow multiple processes to communicate and perform calculations.
- **Process Synchronization**: Understanding the importance of synchronizing processes and managing data exchange in parallel applications.
- **Efficient Data Handling**: Using buffered sends to optimize communication and reduce latency in multi-process applications.

---

## 📂 **Code Structure**

The following files contain the implementations of the tasks:

- **`q1.c`**: MPI program using synchronous send for word manipulation.
- **`q2.c`**: Code implementing master-slave communication for sending a number to slave processes.
- **`q3.c`**: Program where even-ranked processes square the number and odd-ranked processes cube it, using buffered send.
- **`q4.c`**: Chain communication program where processes increment a value and send it back to the root.

---

## 🔗 **Explore Other Labs**

For more exercises and labs, check out the [main repository](https://github.com/adityagarwal15/PCAP-Lab).

---

🚀 **Happy Learning and Coding!** 💻✨
