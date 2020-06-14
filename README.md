# VaniDL

VaniDL is an tool for analyzing I/O patterns and behavior with Deep Learning Applications. It analyzes Darshan Extended traces to extract various I/O patterns with Deep Learning applications. The tool at it core uses dlprofiler which converts Darshan Profiler’s trace into knowledge for analysis. It is designed to provide low-level I/O behavior details to tensorflow applications in HPC. 

**TFLearn features include:**

-   Easy-to-use and understand high-level API for extracting I/O behavior of the applications.
-   Fast prototyping through highly modular data representation through pandas for easy plotting of graphs.
-   Full transparency over profiling data with access to internal data structures such as timeline of applications, aggregation functions, and drill up/down data views.
-   Powerful helper functions to build a visual understanding of how applications perform I/O such as request distributions, file access pattern, and extracting file specific summaries.
-   Easy to use File Summary and Job Summary extractors for understanding the data consumed by Deep Learning Applications
 

## Overview
```python
#Initialize class
profile = DLProfile()
#Load darshan file
status = profile.Load("./run1.darshan")
#Get Job Summary
summary = profile.GetSummary()
```

```python
#Application Timeline of data operations
tl = profile.CreateIOTimeline()
plt.figure(figsize=(20,4))
plt.grid()
plt.plot(tl['time_step'], tl['operation_count']);

```

## Installation

## Getting Started

### Examples

### Documentations

### Usage Examples

## Contributions

## License

MIT License
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzMzMwMjYxMDUsLTE0Mjg0MjcyMDIsMT
A5NzM3MjU5OCwyNTY0OTIyOTQsMTg0OTE4NDUyNF19
-->