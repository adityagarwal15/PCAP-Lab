# 🚀 **Lab 7: String Manipulation in CUDA**
Welcome to **Lab 7** of the **Parallel Computing and Architecture (PCAP)** course! 🌟  
This lab focuses on **parallel string manipulation** using CUDA programming. You'll work with string processing and pattern generation tasks that demonstrate the power of parallel computing for text operations.

---
## 🧑‍💻 **Lab Exercises**

### 1. **Word Occurrence Counter in CUDA**
Implement a CUDA program that counts the occurrences of a specific word in a given sentence using atomic operations.

#### Key Features:
- Uses atomic functions for accurate counting in parallel
- Handles variable-length sentences
- Supports user input for both sentence and target word
- Implements efficient parallel word comparison

#### Example Usage:
```bash
Enter a sentence (max 1023 characters): The quick brown fox jumps over the lazy fox
Enter the word to count: fox
The word 'fox' appears 2 times in the sentence.
```

#### Implementation Details:
- Uses `atomicAdd()` for thread-safe counting
- Each thread processes a potential word position
- Efficient memory management with proper CUDA allocations
- Robust error handling and input validation

---
### 2. **String Pattern Generator in CUDA**
Create a CUDA program that generates a specific pattern from an input string by creating substrings of decreasing length.

#### Pattern Format:
For input string S = "PCAP":
```
PCAP PCA PC P
```

#### Key Features:
- Parallel generation of pattern segments
- Dynamic calculation of output string length
- Handles variable-length input strings
- Clear visual separation between pattern segments

#### Example Usage:
```bash
Enter the input string: PCAP
Input string S: PCAP
Output string RS: PCAP PCA PC P
```

#### Implementation Details:
- Each thread handles specific character positions
- Efficient parallel processing of string segments
- Memory-efficient pattern generation
- Input validation for minimum string length

---
## 🎯 **Learning Objectives**
By completing this lab, you will:
- **Master String Processing in CUDA**: Learn parallel techniques for string manipulation
- **Understand Atomic Operations**: Implement thread-safe counting mechanisms
- **Handle Dynamic Patterns**: Work with variable-length string patterns
- **Manage Parallel Memory**: Practice efficient CUDA memory management
- **Process User Input**: Implement robust input handling in CUDA programs

---
## 📂 **Code Structure**
The lab contains two main CUDA programs:
- **`word_count.cu`**: Implements parallel word counting using atomic operations
- **`string_pattern.cu`**: Generates string patterns with decreasing substring lengths

---
## 🔧 **Requirements**
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 11.0 or higher recommended)
- GCC/G++ compiler
- Basic understanding of CUDA programming concepts

---
## 📚 **Compilation Instructions**

### Word Count Program:
```bash
nvcc word_count.cu -o word_count
./word_count
```

### Pattern Generator:
```bash
nvcc string_pattern.cu -o string_pattern
./string_pattern
```

---
## 💡 **Tips and Best Practices**
1. **Memory Management**:
   - Always free allocated CUDA memory
   - Use proper error checking for memory operations

2. **Input Validation**:
   - Check for empty strings
   - Validate string lengths
   - Handle edge cases appropriately

3. **Performance Optimization**:
   - Choose appropriate block and grid dimensions
   - Minimize memory transfers between host and device
   - Use coalesced memory access patterns

---
## 🔍 **Testing**
Test your implementations with various inputs:
- Different sentence lengths
- Various word patterns
- Edge cases (empty strings, single characters)
- Maximum length strings
- Special characters (if supported)

---
## 📝 **Submission Guidelines**
1. Submit both source files (`word_count.cu` and `string_pattern.cu`)
2. Include a brief report explaining your implementation
3. Document any assumptions or limitations
4. Add comments explaining complex logic

---
## 🤝 **Contributing**
Feel free to suggest improvements or report issues through:
- Lab feedback sessions
- Course forums
- Direct communication with instructors

---
🚀 **Happy Coding and Learning!** 💻✨
