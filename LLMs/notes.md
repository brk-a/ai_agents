# BYO LLM

## 1. build a bigram model
### 1.1. WTF s a bigram?
* a sequence of two adjacent elements from a string of tokens
* tokens are usually letters, syllables or words
* in the context of language and large language models (LLMs), bigrams refer to pairs of consecutive words or characters used to analyse and predict text patterns
* bigram analysis is a basic form of n-gram modelling where n=2
    - helps in understanding language structure, building language models for tasks like speech recognition, text prediction and cryptography
* example: the phrase **"big data"** is a bigram because it is a pair of two adjacent words
### 1.2. how TF  are bigrams used in Language Models (LMs) like GPT?
* bigrams are  used in simple statistical language models to capture local word dependencies
* modern LLMs like GPT rely on more complex transformer architectures considering long-range context; bigrams form a foundational concept showing how word sequences can be probabilistically modelled
* in these simple models, the probability of a word depends only on the immediately preceding word which helps predict or generate text based on observed patterns in training data
### 1.3. how TF to compute bigram probabilities from a corpus
* to compute bigram probabilities, walk through the corpus to count occurrences of each word (unigrams) and each pair of consecutive words (bigrams)
* the bigram probability P(word₂ | word₁) is then estimated as the count of the pair (word₁, word₂) divided by the count of word₁ alone
* formally:
    
    $$ P(w_2|w_1) = \frac{C(w_1, w_2)}{C(w_1)} $$

    where $$C(w_1, w_2)$$ is bigram count
    
    and $$C(w_1)$$ is unigram count
    
* this simple approach enables prediction of the next word given the current one

### 1.4. WTF are the practical limits of bigram models for text generation?
* bigram models are easy to implement but limited because they consider only immediate word pairs, ignoring longer context and linguistic structures
* they often assign zero probability to unseen word pairs making them brittle without smoothing techniques
* increasing `n` (to trigrams or higher) improves context capture but leads to computational challenges and data sparsity
* modern LLMs overcome these limits by using deep learning and attention mechanisms to capture long-range dependencies and richer semantic information for natural text generation

### 1.5. tensors instead of arrays
* in computational linguistics and large language models, arrays are often insufficient for handling complex data structures efficiently
* presenting the tensor...
    - tensors are employed as more scalable and versatile data structures
* context
    - a scalar is a single numerical value, such as an integer or a floating-point number, representing a zero-dimensional tensor
    - a vector is a one-dimensional array of numbers; essentially a list of values without regard to direction for this explanation
    - matrix is a two-dimensional array: for example, a table of numbers arranged in rows and columns
* a tensor generalises this concept to three or more dimensions: for instance, a three-dimensional tensor can be envisioned as a *"matrix of matrices"* or a cube of numbers
    - higher-dimensional tensors extend this pattern to represent even more complex datasets
* tensors allow more efficient representation and manipulation of multi-dimensional data which is crucial in machine learning frameworks such as TensorFlow and PyTorch
* in language models, tensors enable operations on batches of data, word embeddings across contexts and multi-layer processing which simple arrays cannot manage efficiently

### 1.6. tokenisers
* tools used in natural language processing (NLP) to split text into smaller units called tokens
* tokens are the basic building blocks—usually words, subwords or characters—that a language model uses to understand and process language
* types of tokenisers:
    - Word Tokenisers &rarr; break sentences into individual words, often splitting on spaces and punctuation
        - example, "Big data is key" becomes ["Big", "data", "is", "key"].
    - Character Tokenisers &rarr; split text into single characters, useful for languages or tasks requiring fine granularity
        - example, "key" becomes ["k", "e", "y"].
    - Subword Tokenisers &rarr; split words into smaller meaningful pieces (subwords), balancing between word and character tokenisation
        - they help handle unknown or rare words by breaking them into known parts.
* tokenisers work by applying rules or algorithms to segment input text
    - they remove spaces, separate punctuation, and normalise text to prepare it for processing
    - this converts raw text into structured data that models can analyse for patterns and meaning
* effective tokenisation improves model performance on tasks like text classification, translation, and generation. It is a fundamental first step in large language models such as GPT
* tokenisers are the best solution for processing natural language text because they efficiently convert raw, unstructured text into manageable and meaningful units called tokens
     - this is essential for language models and NLP systems that require structured input to analyse and generate human language effectively
    - they solve the core problem of textual ambiguity and complexity by breaking text into consistent pieces, such as words, subwords, or characters, allowing models to handle vocabulary systematically
    - without tokenisation, models would struggle with variable spacing, punctuation, and unseen words
* tokenisers also optimise performance by enabling shared representations of common subword units
    - this reduces vocabulary size and helps models generalise better to new words, improving accuracy and efficiency
* by simplifying text into digestible units, tokenisers are indispensable for training and deploying language models that perform tasks like translation, summarisation, and text generation reliably and at scale
    - this makes them the best practical solution for bridging human language complexity and machine understanding

### 1.7. train and test splits
* train and test splits refer to dividing a dataset into two parts: one used to train the model (the training set) and one used to evaluate or test the model (the test set)
     - this division allows us to assess how well the model has learned to generalise beyond the data it trained on
#### 1.7.1. why TF are they important?
*  prevent Overfitting: Training on all data without testing might make the model just memorize rather than learn patterns
*  measure Generalisation: Testing on unseen data gives a realistic estimate of model performance on new inputs
*  guide Model Improvements: Performance on the test set helps identify whether the model or data needs refinement
#### 1.7.2. how TF do train and test splits work in bigram models?
* split the corpus: The text data is randomly or systematically divided, often with about 70-90% for training and 10-30% for testing
* training set: Used to count unigram and bigram frequencies, from which bigram probabilities are computed
* test set: Used to evaluate the model’s predictions, such as calculating the probability of sequences or measuring perplexity
#### 1.7.3. practical aspects
* random sampling: Ensures the train and test sets are representative of the language
* no overlap: Test data must be distinct and not used in training to ensure unbiased evaluation
* evaluation metrics: Besides accuracy, metrics like perplexity help understand model confidence on test data
#### 1.7.4. simple analogy
* picture learning new words (training), then taking a quiz (test) on words not explicitly studied
* this helps check if you really understood the language pattern or just memorised specific words
### 1.8. premise of a bigram model
* the bigram model is based on the **Markov assumption** that the probability of a word depends only on the immediately preceding word — not on any earlier words in the sequence
* this is a simplification of natural language dependencies but provides a tractable way to model language statistically.
* mathemagically, if we have a sequence of words $$w_1, w_2, \ldots, w_n$$, the joint probability is approximated as:

    $$
    P(w_1, w_2, \ldots, w_n) \approx P(w_1) \prod_{i=2}^n P(w_i | w_{i-1})
    $$

* here, $$P(w_i | w_{i-1})$$ is the bigram conditional probability estimated from data
    - This model only looks one step back, capturing **local context** but ignoring longer-range dependencies
    - It drastically reduces complexity compared to full joint modelling, making computation efficient
    - It reflects the notion that language has some sequential structure but is limited in capturing nuances like syntax and semantics beyond pairs
* the bigram model is often a baseline to understand language statistics or a building block towards more complex models such as trigrams or neural LMs
### 1.9. inputs and targets
* tn training a bigram model, the **input** and **target** have the following relationship:
    - **Input:** Each word $$w_i$$ in the sequence (except the last)
    - **Target:** The next word $$w_{i+1}$$ that follows the input word
* example given the sentence **"The cat sat on the mat"**, we prepare training pairs as:

    | Input ($$w_i$$) | Target ($$w_{i+1}$$) |
    |------------------|-----------------------|
    | The              | cat                   |
    | cat              | sat                   |
    | sat              | on                    |
    | on               | the                   |
    | the              | mat                   |

* the model learns probabilities $$P(w_{i+1}|w_i)$$ by counting how often each target word follows each input word in the training corpus
* mathemagically, if $$V$$ is our vocabulary
    - inputs and targets are drawn from $$V$$
    - for each input $$w_i$$, the model's output is a probability distribution over all words $$w \in V$$ representing $$P(w | w_i)$$
* in practice, when implementing a bigram LM
    - Inputs are often encoded as indices or one-hot vectors representing words
    - Targets are the next words to predict
    - The model parameters are the bigram probabilities stored in a matrix $$B$$ of size $$|V| \times |V|$$ where $$B_{ij} \approx P(w_j | w_i)$$
* this matrix enables quick lookup of conditional probabilities for prediction or text generation

    ```mermaid
        graph LR
            A[w_1: "The"] -->|P(cat | The)| B[w_2: "cat"]
            B -->|P(sat | cat)| C[w_3: "sat"]
            C -->|P(on | sat)| D[w_4: "on"]
            D -->|P(the | on)| E[w_5: "the"]
            E -->|P(mat | the)| F[w_6: "mat"]

            subgraph Bigram Probability Matrix B
                direction TB
                W1[The] --- W2[cat]
                W1 --- W3[sat]
                W2 --- W3[sat]
                W2 --- W4[on]
                W3 --- W4[on]
                W3 --- W5[the]
                W4 --- W5[the]
                W5 --- W6[mat]
            end
    ```

* explanation:
    - the left chain of nodes shows the input word pointing to its target next word with an associated conditional probability
    - the "Bigram Probability Matrix" block symbolises the matrix $$B$$ where each row corresponds to an input word (e.g., “The”) and each column corresponds to a possible next word (e.g., “cat,” “sat”)
    - each cell $$B_{ij}$$ stores the bigram probability $$P(w_j | w_i)$$ estimated from corpus counts
* matrix form
    - if vocabulary $$V = \{ \text{The}, \text{cat}, \text{sat}, \text{on}, \text{the}, \text{mat} \}$$, then the bigram matrix viz:

    $$
    B = \begin{bmatrix}
    P(The|The) & P(cat|The) & P(sat|The) & \cdots \\
    P(The|cat) & P(cat|cat) & P(sat|cat) & \cdots \\
    \vdots & \vdots & \vdots & \ddots
    \end{bmatrix}
    $$

* each row sums approximately to 1 (after smoothing), representing a conditional probability distribution over next words.

### 1.10. batch size hyperparameter
* important in training language models including bigram-based systems and larger neural models
* determines the number of training examples processed simultaneously before the model's parameters are updated
#### 1.10.1. WTF does batch size control?
* memory usage: a larger batch size requires more memory, as more data points are processed in parallel
* gradient estimation: affects how the model estimates the gradient during optimisation. larger batches produce a more stable estimate of the true gradient but require more computational resources
* training stability: larger batches tend to stabilise the training process, reducing the variance of gradient estimates, leading to smoother convergence
#### 1.10.2. how TF does batch size influence training dynamics?
* small batch sizes (e.g., 8, 16):
    - faster updates per iteration, potentially leading to noisier gradient estimates
    - may help escape local minima, adding beneficial stochasticity
    - may require more training epochs for convergence
* large batch sizes (e.g., 128, 256 or higher):
    - more consistent gradient estimates, leading to more stable updates
    - faster epoch times given more data processed per step, but each step is computationally intensive
    - risk of converging to sharp minima, which may generalise less well
#### 1.10.3. trade-offs and practical considerations
* the optimal batch size balances computational feasibility with convergence stability
* modern hardware, like GPUs and TPUs, often set practical upper bounds for batch sizes because of memory constraints
* techniques like gradient accumulation allow effective training with smaller batch sizes on limited hardware, simulating larger batch effects
* empirical tuning is essential; too small may slow learning, too large may cause poor generalisation
#### 1.10.4. in the context of bigram models
* while classic bigram models often operate with counts and probabilities, batch size is more relevant when employing neural networks or probabilistic models trained via optimisation algorithms such as stochastic gradient descent (SGD)
* for neural-based language modules, choosing an appropriate batch size directly impacts training efficiency and effectiveness
### 1.11. switching from CPU to CUDA
* as the scale of data and model complexity grows, even a simple bigram model can benefit from **hardware acceleration**
* in particular, moving computations from the **CPU (Central Processing Unit)** to the **GPU (Graphics Processing Unit)** using **CUDA (Compute Unified Device Architecture)** can yield significant performance gains when implemented correctly.
#### 1.11.1. what the actual f is CUDA?
* a parallel computing platform and programming model developed by **NVIDIA**, allowing general-purpose computations to be executed on GPUs rather than being limited to CPUs
* conceptually, the CPU is a **few very powerful cores**, while a GPU is **thousands of smaller, simpler cores** designed to perform the same operation on many data elements simultaneously — an architecture ideal for **matrix–tensor operations** that dominate machine learning workloads
* CUDA exposes the GPU’s parallelism through extensions to programming languages like C++, Python (via PyTorch, TensorFlow, CuPy) and Fortran
* formally, if a CPU executes operations serially on a dataset $f\{X\}$, the time complexity can be approximated as:

$$
T_{CPU} = O(N)
$$

where ( N ) is the number of elements

* in contrast, a CUDA kernel running on a GPU can process ( p ) elements concurrently (where ( p ) is the number of CUDA cores), giving:

$$
T_{\text{GPU}} = O\left(\frac{N}{p}\right)
$$

yielding a **speed-up factor** of approximately $ \approx \frac{N}{N/p} = p $, though in practice the benefit is sublinear due to data transfer and synchronisation overheads
#### 1.11.2. the computational case for GPUs in bigram models
* even though bigram models are relatively simple, when implemented using **tensorised operations** (as in PyTorch), computations scale quadratically with vocabulary size:

$$
B \in \mathbb{R}^{|V| \times |V|}
$$

* for a vocabulary of 50,000 tokens, this matrix contains $ 2.5 \times 10^9 $ elements; matrix operations such as:

$$
\mathbf{P}_{\text{next}} = \text{softmax}(\mathbf{x} W)
$$

where $ \mathbf{x} $ is the input tensor and ( W ) the bigram weight matrix, involve dense linear algebra—an area where GPUs excel due to their high **FLOPs (Floating Point Operations per Second)** throughput
* the computational bottlenecks include:
    - **Tensor multiplication** – matrix–vector or matrix–matrix operations; highly parallelisable
    - **Softmax normalisation** – involves exponentiation and division across large vectors; can be GPU-parallelised
    - **Gradient updates (if training a neural variant)** – backpropagation involves matrix transposes and dot products, also parallelisable

#### 1.11.3. moving from CPU to CUDA in PyTorch
* in PyTorch, switching from CPU to CUDA is typically a one-line operation, but conceptually it involves changing the device on which tensors are allocated
* example pseudocode:

    ```python
        import torch

        # Automatically select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vocabulary size and bigram weight matrix
        V = 50000
        B = torch.randn(V, V, device=device)  # allocate directly on GPU

        # Sample batch of indices
        inputs = torch.randint(0, V, (64,), device=device)
        targets = torch.randint(0, V, (64,), device=device)

        # Lookup and compute predictions
        logits = B[inputs]  # matrix row lookup (GPU)
        probs = torch.softmax(logits, dim=1)
        loss = -torch.log(probs[torch.arange(64), targets]).mean()

        loss.backward()  # gradients computed via CUDA kernels
    ```

* when using the `device` argument, all operations, including forward and backward passes, are executed directly on the GPU using CUDA kernels
#### 1.11.4. data transfer costs and memory hierarchy
* while GPUs deliver **massively parallel throughput**, they also introduce new constraints in **data transfer and memory management**
* the performance advantage can be negated if the model frequently moves data between CPU and GPU.

* **Memory hierarchy (simplified):**

| Memory Type           | Typical Bandwidth | Access Latency | Scope       |
| --------------------- | ----------------- | -------------- | ----------- |
| GPU registers         | ~20 TB/s          | <10 ns         | per-thread  |
| GPU shared memory     | ~5 TB/s           | <100 ns        | per-block   |
| GPU global memory     | ~1 TB/s           | ~500 ns        | device-wide |
| CPU main memory (RAM) | ~50–100 GB/s      | ~100 ns        | host        |
| PCIe bus (CPU–GPU)    | ~16–32 GB/s       | >1 μs          | transfer    |

* **Implication:** minimise host–device transfers. For bigram models, load the corpus, tokeniser, and tensor initialisation once on the GPU, and keep computation there
#### 1.11.5. numerical precision and stability
* CUDA supports multiple numeric precisions:
    - **FP32 (single precision):** default, good balance between speed and accuracy
    - **FP16 (half precision):** halves memory footprint; doubles throughput on compatible hardware but risks underflow/overflow in small probability computations (e.g. during softmax)
    - **BF16 (bfloat16):** alternative to FP16 that preserves exponent range, improving stability
    - **TF32:** hybrid precision used on Ampere GPUs for matrix multiplication acceleration
* in probabilistic bigram models, the risk of **underflow** is real due to small $ P(w_i | w_j) $ values; it is often numerically safer to work in **log space**:

$$
\log P(w_2 | w_1) = \log C(w_1, w_2) - \log C(w_1)
$$

* GPU kernels can handle this transformation efficiently
#### 1.11.6. parallelisation and stochastic optimisation
* when training a neuralised bigram model using **stochastic gradient descent (SGD)** or **Adam**, CUDA parallelises the gradient computations across the batch
* for batch size ( b ), vocabulary size ( V ) and embedding dimension ( d ):

$$
O(b \times V \times d)
$$

* operations per iteration are parallelised; this reduces wall-clock time significantly, especially for large ( V ) and ( d )
* example:
    * CPU (8 cores): ~50 GFLOPs/s
    * GPU (RTX 4090): ~80 TFLOPs/s
    → theoretical **speedup ≈ 1600×** (practical ≈ 30–100× after overheads)
#### 1.11.7. software engineering principles for GPU migration
* when migrating to CUDA, adhere to sound engineering practices:
    - **Device abstraction:** define a `device` variable and pass it consistently through your code: do not hardcode `"cuda:0"`
    - **Batch computation:** ensure that operations are vectorised; avoid Python loops over tokens
    - **Profiling and optimisation:** use `torch.cuda.profiler`, `nvprof`, or Nsight Systems to identify kernel bottlenecks
    - **Memory efficiency:** release unused tensors via `del` and `torch.cuda.empty_cache()`
    - **Determinism:** some GPU kernels are non-deterministic; set `torch.backends.cudnn.deterministic = True` if reproducibility matters
#### 1.11.8. probabilistic and statistical considerations
* moving computation to CUDA does not alter the **probabilistic semantics** of the model—only the numerical performance, however, stochastic training on GPUs can introduce small **floating-point non-determinisms**, potentially affecting repeatability of random seeds
* in expectation, the estimated conditional probabilities:

$$
\hat{P}(w_j | w_i) = \frac{C(w_i, w_j)}{C(w_i)}
$$

remain identical; only the computational path changes
* GPU acceleration primarily affects:
    - **Speed of computing counts and normalisations** (when done as batched tensor operations)
    - **Parallel computation of losses or perplexities** across test batches

#### 1.11.9. illustrative example: CPU vs CUDA timing

    ```python
        import torch, time

        V = 10000
        B_cpu = torch.randn(V, V)
        B_cuda = B_cpu.to("cuda")

        inputs = torch.randint(0, V, (512,))
        targets = torch.randint(0, V, (512,))

        # CPU timing
        t0 = time.time()
        probs_cpu = torch.softmax(B_cpu[inputs], dim=1)
        loss_cpu = -torch.log(probs_cpu[torch.arange(512), targets]).mean()
        t_cpu = time.time() - t0

        # CUDA timing
        torch.cuda.synchronize()
        t1 = time.time()
        probs_gpu = torch.softmax(B_cuda[inputs.to("cuda")], dim=1)
        loss_gpu = -torch.log(probs_gpu[torch.arange(512), targets.to("cuda")]).mean()
        torch.cuda.synchronize()
        t_gpu = time.time() - t1

        print(f"CPU: {t_cpu:.4f}s, GPU: {t_gpu:.4f}s, Speed-up: {t_cpu/t_gpu:.1f}×")
    ```

* typical output on a modern GPU might be:

    ```plaintext
        CPU: 2.1342s, GPU: 0.0347s, Speed-up: 61.5×
    ```

#### 1.11.10. summary — from serial to parallel thought
* switching from CPU to CUDA is not just a hardware migration; it is a **computational paradigm shift** from:
    - **serial** to **parallel** data processing
    - **cache-optimised loops** to **vectorised tensor kernels**
    - **general-purpose control flow** to **mathematical throughput maximisation**
* for bigram models, CUDA may seem overkill but it lays the foundation for scaling up to **neural language models** where matrix–tensor operations dominate training
* in statistical terms, CUDA merely preserves the same estimator — but performs it at a rate limited more by **memory bandwidth** than by **algebraic complexity**

> **long story short:** switching from CPU to CUDA turns your bigram model from a statistical toy into a computational experiment in high-throughput probabilistic inference: same maths, different universe
### 1.12. embedded vectors
* 