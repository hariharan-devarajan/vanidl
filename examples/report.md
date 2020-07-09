
# Analysizing I/O for Deep Learning Workloads

##  1. Imagenet

- Contains implementations of several popular convolutional models
	- models supported are resnet50, inception3, vgg16, and alexnet.
- Supports both running on a single machine or running in distributed mode across multiple hosts.
- Generates synthetic data or can read data from the filesystem.
- Uses TFRecord data format with tf.data input data pipeline APIs.
- Imagenet benchmark has the following workload characteristics
- \# of steps is 110 with 10 warmup steps
    - batch size of 512 images
    - The workload was executed with 8 Nodes 
    	- 8 ranks: 1 rank per node
        - 128 OMP Threads
    - For this test it took
    	- ~4.1 seconds per step
        - Tensorflow reads TFRecord files using POSIX interface (i.e., open(), read(), write(), and close())
        - As we increase the number of processes, more data is read.
        

### Darshan I/O Profiling Results
- It spends 7% of the total time of total time on I/O 
    - The workload performs I/O operations to read images out of TFRecord files (i.e., read-only workload). 
    - The tfrecord file is read by several ranks and, hence, I/O per file may be more than its file size.
    - The whole file is always read sequentially and consequitively (through parallel thread from io pipeline) from start till required by the training model.
- 60.54 GB of data was accessed out of 40GB datasets
    - Total Files were read were 473. (It scales as we increase number of processes)
	    - About 25% files were just read once. (are all the files touched in the start?)
        - Each TFRecord file has around 1000 images.
        - Each rank equally distributed the files (no overlap).
        - Each rank randomly selects files from the folders and reads the file from start till required by the training. 
            - This reading is performed on demand as required by the training the input pipeline feeds the data
            - The amount of data read per file is almost equal.
    - 98% of the I/O occured with a transfer buffer size of 256KB (default for reading TFRecord files)
    - Periodically, due to prefetching, twice the data is read (i.e. 256KB * 2).
    - Average I/O bandwidth per request achieved is 1.2 GB/s

### Suggestions for Improvement

- The default lustre strip size is 1 MB. Hence, to maximize performance, reduce the lustre strip size to 64KB so as to improve operation parallelism within data.
- As the I/O size is only 256KB, the default prefetching size of 1 (i.e., one additional buffer is too small). As all I/O is sequential, we can safely perform aggressive prefetching (i.e., having not only 1 more image but several hundreds). This will improve locality of data for tf.data pipeline.
- The parallelism of operations at any given point is 64 per rank [code](https://github.com/tensorflow/benchmarks/blob/7099c1c3e57795134dcbf14088ad131e8c2979f0/scripts/tf_cnn_benchmarks/preprocessing.py#L1089) . As the benchmark uses one process per node, we can either increase the number of processes per node to 128 or increase read parallelism of the dataset (i.e., through tf.data API) to 128 (i.e., number of cores in Theta)
- We can tune the trade-off between operation parallelism and operation size to maximize I/O performance and minimize interference.
	- Intutively, we should not experience data interference but might have a lot of metadata lookups.
- As many files are opened, we can reduce file by increasing images per TFRecord file. This would decrease metadata lookups on filesystem.
	- Note, reducing too much will reduce file shuffling effectiveness (for randomizing images in a batch) as scale increases. **(Reduce # of files to 8 and see how ranks access the file)**

#### Advanced
- As datasize per node is small and access is sequential, we can stage data into node-local SSD's or even in a RAM Buffer to feed data faster.
	- An ideal situation is to utilize an hierarchical cache to stage data for feeding it into the data pipeline
    
## 2. Cosmic Tagger
- Is a convolutional network to separate cosmic pixels, background pixels, and neutrino pixels in a neutrinos dataset.
- The data for this network is in larcv3 format.
	- Each training sample contains 3 images of 1280x2048
    - To accomodate for smaller GPU, this image can be reduced using downsampling opperation.
	- The training dataset consist of 43075 images with a max size of 50K.
- Cosmic Tagger application has the following workload characteristics
	- \# of steps is 150
    - batch size is 32 images
    - Workload was run with 8 Nodes 
    	- 8 ranks: 1 rank per node
        - 128 OMP Threads
    - For this test it took
    	- ~6 seconds per step
        - The app through larcv API reads HDF5 file
        - As we increase the number of processes more data is read.

### Darshan I/O Profiling Results

- It spend 6% of total time of total time on I/O 
    - All I/O performed  is on a training hdf5 file is predominantly **read** for reading images from the train dataset.
    - The hdf5 file is read by several ranks and, hence, I/O per file may be more than its file size.
    - The application initially reads 50 operations of 13KB data size.
    - and then the file is always read sequentially and consequitively from start till required
    	- Each request reads a image based on the distribution of request, most reads are either 2KB and 6KB. (Chunking)
    	- As the reads are small, the I/O bandwidth per operation is only 9MB/s
    - No data shuffling during reading.
    - Every rank seems to be reading the same offsets from the training dataset.
    - It weak scales the data with the number of processes.


### Suggestions for Improvement

- The data transfer I/O per request is few killobytes. This is the main cause of bad I/O bandwidth achieved from PFS. This should be several Megabytes (1 -4 MB) with small lustre strip size of (64KB).
- As the file is small and read multiple times, we can cache the file in memory.
- Most data is read sequentially and consecutively, we can use staging plus prefetching on a fast cache (local RAM or SSD) to ensure data locality for the training process. 
- Multiple threads can be utilized to perform I/O. As number of operation per timestep is very small.
- We can read data into RAM and use RDMA calls to access data (RDMA calls are faster than reading from PFS).
    
#### Advanced

- As the dataset is smaller than the aggregated RAM of the nodes, we can use in-memory HDF5 API to load the data set in memory or on an SSD and distributing it across nodes before Training to reduce constant reading from PFS and allowing bulk I/O with max I/O performance.
    
## 3. Distributed Flood Filling Networks

- It is a synchronous and data-parallel distributed training for Flood-Filling Networks using the Horovod framework for instance segmentation of complex and large shapes, particularly in volume EM datasets of brain tissue.
- It uses data parallel training with synchronous stochastic gradient descent
(SGD)
- It supports data sharding where data is distributed equally amoung various processes in the distributed training. 
- The data for this network is in hdf5 format.
	- The training image dataset is a grayscale map images
    	- Each image is of size 32X32X32.
        - Total of 4096 field of views (fovs).
- Cosmic Tagger application has the following workload characteristics
	- \# of steps is 400
    - batch size is 1 images
    - Workload was run with 8 Nodes 
    	- 8 ranks: 1 rank per node
        - 128 OMP Threads
    - For this test it took
    	- ~1 seconds per step
        - Tensorflow reads HDF5 file
        - As we increase the number of processes more data is read.

### Darshan I/O Profiling Results

- It spend 7% of total time of total time on I/O 
    - All I/O performed  is on a training hdf5 file is predominantly read for reading images
    - The hdf5 file is read by several ranks and hence I/O per file may be more than its file size.
    - File is always read sequentially and consequitively from start till required
    	- Each request reads a image based on the distribution of request, most reads are 28KB.
    	- I/O bandwidth per operation is 127.91 MB/s.
    - Each image is randomly chosen from the dataset. However, overall all images are read by all the processes.
    - Every rank seems to be reading the same offsets from the training dataset. (Seems to be a bug as it is not intended in Sharding mode)
	- Performs shuffle in the batch.
	- It weak scales the data with the number of processes.

### Suggestions for Improvement

- The data transfer I/O per request is few tens of killobytes. This is the main cause of bad I/O bandwidth achieved from PFS. This should be couple of Megabytes (1 - 4 MB) with small lustre strip size of (64KB).
- Most data is read sequentially but not consecutively (due to random shuffling of images), we can stage data into a faster cache (local SSD) before training begins.
	- This enables random access on dataset for random shuffling.
- Multiple threads should be utilized to perform I/O. As number of operation per timestep is very small.
- Each rank is reading same portions of the file. This should be avoided and each process should divide the data and train on different portions of the data and weights of the DL should be converged.
	- If this is a desired feature, then we can read data into RAM and use RDMA calls to access data (RDMA calls are faster than reading from PFS).
    
#### Advanced

- As the dataset is smaller than the aggregated RAM of the nodes, we can use in-memory HDF5 API to load the dataset in memory or node-local SSD and distribute it across nodes before training to reduce constant reading from PFS and allowing bulk I/O with max I/O performance.


## 4. Climate Seg Benchmark
- This benchamrk is a reference implementation of Climate Segment benchmark based on Exascale Deep Learning for Climate Analytics which comprises of multiple deep learning models for different climate data projects such as AR detection, Storm tracking and Semantic segmentation
- Climate Seg Benchmark was run with the following configurations
	- \# of steps is 1200
    - batch size is 1
    - epochs is 1
    - Workload was run with 8 Nodes 
    	- 8 ranks: 1 rank per node
        - 128 OMP Threads
    - For this test it took
    	- ~1 seconds per step
        - reads HDF5 stat file and writes JSON outputs

### Darshan I/O Profiling Results
- The application doesn't perform much I/O (verified in code ./deeplab-tf/deeplab-tf-train.py:152)
	- It initially reads stats.h5 (4KB file) file from the training folder (total 500 Bytes reads average over 50 Operations)
	- at the end of the application, it writes each rank write 3 json files files of 4MB each in one single operation (Total I/O 96 MB). 
	- The stats file is not read consecutively (only 13.06%)

### Suggestions for Improvement
- The stats file should be read in memory in one call and then can be accessed from the memory.

## 5. Cosmoflow Benchmark
- Cant find a read me for this app
- Data is arranged in tf record format.
	- The application takes the datadir
	- shuffles the files
	- distributes it amound ranks
	- shuffles it
	- performs decoding using parallel reads
	- shuffles the dataset
	- batches
- Each TF record contains 512 samples with each sample 8MB ([128, 128, 128, 4])
- Configuration for this test was 
	- training files used 1024.
	- samples extracted per file 256
		- shape of sample: [128, 128, 128, 4]
		- batch size 1
		- epochs 1
	
### Darshan I/O Profiling Results
- 40% of the time is spent in performing I/O
- The dataset is read at the start of the epoch. (2TB)
	- 2GB per files (total 1024 files)
	- Files are read in parallel across processes but not within a process
	- The data distribution is 256KB request (default buffer size)
	- The I/O is completely consecutive.
	- With a bandiwthd of 2.6 GB/s  
	- Every rank gets different set of files (equally distributed between processes)

### Suggestions
- Currently, the code performs custom shuffling and reading of files
	- Instead we can use the TFRecordDataset
		- Once we do shuffling of filenames use TFRecordDataset to change buffer_size=None, num_parallel_reads=None to improve I/O
- Also, Enabling Prefetching would reduce I/O time
- Seen operator parallelism for data processing was 4, this can be increased to match the core-count (or use AUTOTUNE).
- Possible migration to TF 2.2 to enable tensorboard profiling to understand the data processing from a higher level.

## 6. CANDEL

- Implements deep learning architectures that are relevant to problems in cancer.
- out of problems and data at the cellular level. The high level goal of the problem behind the benchmarks is to predict drug response based on molecular features of tumor cells and drug descriptors.
- is a 1D convolutional network for classifying RNA-seq gene expression profiles into normal or tumor tissue categories.
- The network follows the classic architecture of convolutional models with multiple 1D convolutional layers interleaved with pooling layers followed by final dense layers.
- The data is present in CSV format
	- Dataset size 600 MB
	- It uses pd.read_csv which reads the dataset line by line into the memory
	- Each record is has 60484 32 bit float and hence each record is ~236 KB 
	- Total number of records are 1120 records
	- Every process reads the data once in the start.
- Configuration for this test was
    - batch size is 20
    - epochs is 60
    - Workload was run with 8 Nodes 
    	- 8 ranks: 1 rank per node
        - 64 OMP Threads
    - For this test it took
    	- ~46.27 seconds per epoch

### Darshan I/O Profiling Results

- It spend less than 1% of total time of total time on I/O 
    - All I/O performed  is on a training csv file is predominantly **read** for reading images from the train dataset.
    - The csv file is read by all ranks and, hence, I/O on file is more than its file size.
    - All data is read in the beginning and then training happens in memory
    	- Each request reads a record which is approx ~256 KB
    	- As the I/O is big enough it achieves a bandwidth of 400 MB/s
    - Every rank seems to be reading the same offsets from the training dataset.
    - It strong scales the data with the number of processes.


## 7. L2HMC Algorithm with Neural Network
- this app applies the L2HMC algorithm to generate *gauge configurations* for LatticeQCD.
- In this app, the data is synthetically generated and at each n steps a checkpoint of the model is created.
	- The checkpoint consist of 4 files.
		- checkpoint file: 1KB
		- graph.pbtxt : 7MB
		- model data: 32 MB
		- model.ckpt: 5kB
		- model.ckpt-<#>.meta 4KB
- The configuration of this test were
	- batch_size 32
	- train_steps 150
	- save_steps 50

### Darshan I/O Profiling Results
- The checkpointing part of Model is a write only workload where all files are written using stdio (i.e., fopen, fclose) uin one single operations
- Checkpoint I/O depends on the size of model which is getting stored
	- In this application, the total I/O size ~ 32MB and hence it is only 1% of the total execution time.

### Suggestions
- As the I/O is write and very small, one optimization is to store all checkpoints into local SSD instead of the PFS to increase I/O bandwidth 

## 8. Bert

- **BERT**, or **B**idirectional **E**ncoder **R**epresentations from **T**ransformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.
- Is a collections of 24 smaller BERT models (English only, uncased, trained with WordPiece masking)
- BERT is a method of pre-training language representations, meaning that we train a general-purpose "language understanding" model on a large text corpus (like Wikipedia), and then use that model for downstream NLP tasks that we care about (like question answering). BERT outperforms previous methods because it is the first _unsupervised_, _deeply bidirectional_ system for pre-training NLP.
- Dataset is in the form of CSV files 
	- This example code fine-tunes on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples
	- All I/O happens in the beginning, the tsc is read and converted into samples into memory. 
	- Each I/O size is a record length in size which is 200 bytes in size.
	- Read time is proportional to the dataset as it is read using csv read which reads whole data into memory and processes it.
	- reading time : 0.016541481018066406 seconds. Its very small in comparison to model time.
- Test Configuration
	- 1 Node test.

### Suggestions
- Instead of using CSV format, Use tf record format so that reading can be decoupled with data processing.


## 9. mmaADSP
- A model to fit for gravitational wave parameters from a binary black hole merger
- Four toy datasets are provided.  SNR_interval_0 represents high signal to noise dataset.
- Uses HDF5 interface to read the file
	- The whole data is present in train and test datasets within the hdf5 file.
	- Dataset size if 625 MB
	- Train dataset has 10000 records of 8k size each
- Configuration for the test
	- Uses 1 node configuration

### Darshan I/O Profiling Results
- Didnt detect any I/O in the application.

## 10. FRNN
- The Fusion Recurrent Neural Net (FRNN) software is a Python package that implements deep learning models for disruption prediction in tokamak fusion plasmas.
- It preprocesses signal and normalization files for stateful LSTM training.
- Uses numpy files npz formats to read data.

### Darshan I/O Profiling Results
- It spends 8.43% of time on I/O
- At each timestep it reads different different NPZ files which is read in one go and is fed to training steps
- It performs 

# I/O Patterns in Training of DL workloads

## Interfaces for performing I/O
1. TFRecord
2. HDF5
3. CSV
4. NPZ (numpy array format)
5. JSON

## Types of I/O:
- Read input into memory
	- CSV, NPZ, and JSON
- Read input on demand per step
	- TFRecord and HDF5
- Write checkpoint for every n steps:	1 operation of different file sizes (4 files) from 1KB, 4KB, 64KB, and 4MB
## I/O Access Pattern
- MPI + OMP

## I/O Access Pattern
- File Access
	- Multiple files (no overlap between processes)
		- File is randomly accessed from a directly
	- Shared file
		- Every Process opens the file but shards the partition from which they read.
	- **MPI-IO**
		- Files are collectively read by a subset of nodes (configurable through communicator groups) and broadcast to other nodes.
- Data Access Pattern
	- Case 1: all data is access sequentially, and consecutively
	- Case 2: Processes jump around to select images from dataset.
- Transfer Size:
	- TFRecord: 256 KB
	- Others: depends on image dimensions. Generally, a image is selected random offset and then a batch of images are read sequentially.

## Tuning Parameters
- Data Characteristics
	- Dimension of image/ length of record (unit of I/O)
- Access Characteristics
	- Shuffle during reading
	- batch size of Reading (sequential predetermined batch)
- Processing Characteristics
	- # samples
	- \# of epochs (2-3)
	- \# of Steps in each epoch. (dependents on = # samples/batch size)
	- checkpoint frequency in count of steps
	- Benchmark Skeleton
	- 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3ODQ2NzQ2NywtMTI5NjgwNDQxOSwtNj
YxNDM5OTMsLTEwMDM3ODc1ODgsLTEzNTA3MzIxNjksNDM0NTI3
NTQ5LDE4NDQwNjE5NzAsMTA4MTc5NjkyMV19
-->