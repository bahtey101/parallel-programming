# Repository for Laboratory Works on Parallel Programming in C++ using OpenMP and MPI

## Description
This repository contains solutions to laboratory works completed as part of a course on parallel programming in C++. The main focus is on using two parallelism models: OpenMP for multithreaded programming and MPI (Message Passing Interface) for distributed programming. Each laboratory work includes source code and a brief description of the implemented algorithms.

## Laboratory Works
### Lab 2: Synchronization Methods
In this lab, various synchronization methods were compared to identify the most efficient one. For OpenMP, three different synchronization techniques were analyzed: Atomic, Critical, and Reduction. Each method's performance was evaluated based on execution time and resource utilization. Additionally, for MPI, point-to-point communication methods and collective operations such as Reduce and Broadcast were tested.

### Lab 3: Scheduling Techniques
This lab focused on the use of scheduling techniques in OpenMP and the implementation of data distribution methods in MPI to optimize parallel execution. In OpenMP, different scheduling policies were examined, including Static, Dynamic, and Guided. Each scheduling method was implemented in a parallel program, and the impact on performance was measured. Additionally, MPI methods such as Scatter, Scatterv, Gather, and Gatherv were utilized for distributing and collecting data among processes. The goal was to understand how different scheduling strategies and data distribution techniques affect load balancing and overall execution time in parallel applications. 

### Lab 4: CUDA
In this lab, the focus was on developing parallel programs using CUDA to harness the computational power of GPUs. Key concepts such as thread hierarchy, memory management, and kernel optimization were explored.