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