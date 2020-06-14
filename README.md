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
More examples are [here](https://github.com/hariharan-devarajan/dlprofiler/tree/master/examples)

## Installation

### Requirements
- numpy==1.18.5
- pandas==1.0.4
- h5py==2.10.0
- tensorflow~=2.2.0

**VaniDL Installation**

To install VaniDL, the easiest way is to run

For the bleeding edge version (recommended):

```bash
pip install git+https://github.com/hariharan-devarajan/dlprofiler.git
```
For the latest stable version:
```bash
pip install vanidl
```
Otherwise, you can also install from source by running (from source folder):
```bash
python setup.py install
```
### On Theta
```bash
module load VaniDL
```
## Getting Started
See _[Getting Started with VaniDL](http://vanidl.github.io/getting_started)_ to learn about VaniDL basic functionalities or start browsing _[TFLearn Tutorials](http://tflearn.org/tutorials)_.

### Examples

### Documentations

### Usage Examples

## Contributions

## License

MIT License
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTY0MzU2MDk4MSwtMTMyMTg5NDY3OSwtMT
MzMzAyNjEwNSwtMTQyODQyNzIwMiwxMDk3MzcyNTk4LDI1NjQ5
MjI5NCwxODQ5MTg0NTI0XX0=
-->