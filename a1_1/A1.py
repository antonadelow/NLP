
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

nltk.download('punkt_tab')
###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).
    counter = Counter()
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counter.update(tokenize_fun(line))
            
    # Assign special tokens first
    specials = [pad_token, unk_token, bos_token, eos_token]
    str_to_int = {t: i for i, t in enumerate(specials)}
    
    # Calculate how many normal words we can include
    if max_voc_size is not None:
        rem_size = max_voc_size - len(specials)
        most_common = counter.most_common(rem_size)
    else:
        most_common = counter.most_common()
        
    for word, _ in most_common:
        if word not in str_to_int:
            str_to_int[word] = len(str_to_int)
            
    int_to_str = {i: w for w, i in str_to_int.items()}
    
    return A1Tokenizer(str_to_int=str_to_int, int_to_str=int_to_str, tokenize_fun=tokenize_fun, 
                       model_max_length=model_max_length, pad_token=pad_token, unk_token=unk_token, 
                       bos_token=bos_token, eos_token=eos_token)

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, str_to_int, int_to_str, tokenize_fun, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
            # TODO: store all values you need in order to implement __call__ below.
            self.str_to_int = str_to_int
            self.int_to_str = int_to_str
            self.tokenize_fun = tokenize_fun
            
            self.pad_token_id = self.str_to_int[pad_token]     # Compulsory attribute.
            self.unk_token_id = self.str_to_int[unk_token]
            self.bos_token_id = self.str_to_int[bos_token]
            self.eos_token_id = self.str_to_int[eos_token]
            self.model_max_length = model_max_length # Needed for truncation.

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
            
            Args:
                texts:           The texts to tokenize.
                truncation:      Whether the texts should be truncated to model_max_length.
                padding:         Whether the tokenized texts should be padded on the right side.
                return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

            Returns:
                A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        input_ids = []
        attention_mask = []
        max_len = 0
        
        for text in texts:
            tokens = self.tokenize_fun(text)
            
            # Truncate the raw text first to preserve special tokens
            if truncation and self.model_max_length is not None:
                max_tokens = self.model_max_length - 2 
                tokens = tokens[:max_tokens]
                
            ids = [self.bos_token_id] + [self.str_to_int.get(w, self.unk_token_id) for w in tokens] + [self.eos_token_id]
                
            input_ids.append(ids)
            if len(ids) > max_len:
                max_len = len(ids)

        if padding:
            for i in range(len(input_ids)):
                pad_len = max_len - len(input_ids[i])
                attention_mask.append([1] * len(input_ids[i]) + [0] * pad_len)
                input_ids[i] = input_ids[i] + [self.pad_token_id] * pad_len
        else:
            for i in range(len(input_ids)):
                attention_mask.append([1] * len(input_ids[i]))

        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.
        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=10000, embedding_size=256, hidden_size=512, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)

        self.all_tied_weights_keys = {}

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.LSTM(config.embedding_size, config.hidden_size, batch_first=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)

        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)


    def forward(self, input_ids, labels=None):
        """The forward pass of the RNN-based language model.
        
           Args:
             - input_ids:  The input tensor (2D), consisting of a batch of integer-encoded texts.
             - labels:     The reference tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             A CausalLMOutput containing
               - logits:   The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
               - loss:     The loss computed on this batch.               
        """
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        logits = self.unembedding(rnn_out)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tensors for CrossEntropyLoss
            loss = self.loss_func(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutput(logits=logits, loss=loss)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        #assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)

        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
        #       labels = input_ids with padding replaced by -100
	    #       put input_ids and labels onto the GPU (or whatever device you use)
        #       apply the model to input_ids and labels
        #       get the loss from the model output
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward()
        #       optimizer.step()

        for epoch in range(args.num_train_epochs):
            self.model.train()
            for batch in train_loader:
                # Handle varying dataset formats from HuggingFace
                texts = batch['text'] if isinstance(batch, dict) and 'text' in batch else batch
                
                # Preprocessing and forward pass
                encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
                input_ids = encodings['input_ids'].to(device)
                
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                labels = labels.to(device)
                
                # Forward pass
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
                
                # Backward pass and model update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)


###
### Part 5. Evaluation and analysis.
###

def predict_next_word(model, tokenizer, text, k=5, device='cpu'):
    """ Task 5.1: Predict the top-k next words for a given prefix text.
    
        Args:
            model:      The trained A1RNNModel.
            tokenizer:  The trained A1Tokenizer instance.
            text:       The prefix string (e.g., "She lives in San").
            k:          The number of top predictions to return.
            device:     The device ('cuda', 'mps', or 'cpu') the model is on.
    """
    model.eval()
    model.to(device)
    
    # Tokenize without padding since it's a single sentence prefix
    encodings = tokenizer([text], truncation=True, padding=False, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # Shape: (1, sequence_length, vocab_size)
        
        # We target the logits at the final token position to predict what comes next
        next_token_logits = logits[0, -2, :]        
        # Extract the values and indices of the top-k highest-scoring tokens
        scores, indices = torch.topk(next_token_logits, k)
        
        predictions = []
        for score, idx in zip(scores, indices):
            word = tokenizer.int_to_str.get(idx.item(), '<UNK>')
            predictions.append((word, score.item()))
            
    return predictions


def compute_perplexity(model, val_dataset, tokenizer, batch_size=32, device='cpu'):
    """ Task 5.2: Compute the perplexity score over the validation dataset.
    
        Args:
            model:          The trained A1RNNModel.
            val_dataset:    The validation dataset.
            tokenizer:      The trained A1Tokenizer instance.
            batch_size:     The batch size to use during evaluation.
            device:         The evaluation device.
    """
    model.eval()
    model.to(device)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            texts = batch['text'] if isinstance(batch, dict) and 'text' in batch else batch
            
            encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels = labels.to(device)
            
            outputs = model(input_ids, labels=labels)
            
            # The cross-entropy loss returned by the model is averaged across non-masked tokens.
            # To get an exact average across the entire corpus, we accumulate weighted by active tokens.
            # Slicing [:, 1:] ensures we don't count the dropped <BOS> tokens
            num_active_tokens = (labels[:, 1:] != -100).sum().item()
            
            if num_active_tokens > 0:
                total_loss += outputs.loss.item() * num_active_tokens
                total_tokens += num_active_tokens
                
    mean_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = np.exp(mean_loss)
    
    return perplexity


def nearest_neighbors(emb, voc, inv_voc, word, n_neighbors=5):
    """ Task 5.3: Compute the nearest neighbors in the embedding space using cosine similarity.
    
        Args:
            emb:          The nn.Embedding layer of your language model (model.embedding).
            voc:          The string-to-integer mapping (tokenizer.str_to_int).
            inv_voc:      The integer-to-string mapping (tokenizer.int_to_str).
            word:         The target query word string (e.g., "sweden").
            n_neighbors:  Number of neighbors to fetch.
    """
    if word not in voc:
        print(f"Word '{word}' not found in vocabulary.")
        return []
        
    # Look up the embedding for the test word
    test_emb = emb.weight[voc[word]]
    
    # Use a cosine similarity function to find the most similar words
    sim_func = nn.CosineSimilarity(dim=1)
    cosine_scores = sim_func(test_emb, emb.weight)
    
    # Find the positions of the highest cosine values
    near_nbr = cosine_scores.topk(n_neighbors + 1)
    topk_cos = near_nbr.values[1:]
    topk_indices = near_nbr.indices[1:]
    
    # Map word indices back to strings, skipping the first position (the query word itself)
    return [(inv_voc[ix.item()], cos.item()) for ix, cos in zip(topk_indices, topk_cos)]


if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import TrainingArguments
    import os

    # --- Configuration ---
    TRAIN_FILE = 'a1_1/train.txt'  
    VAL_FILE = 'a1_1/val.txt'      
    OUTPUT_DIR = 'a1_1/trainer_output'
    TOKENIZER_FILE = 'a1_1/a1_tokenizer.pkl'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets (Task 2.1)
    print("Loading datasets...")
    dataset = load_dataset('text', data_files={'train': TRAIN_FILE, 'val': VAL_FILE})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')

    # --- Condition: Load or Train ---
    if os.path.exists(OUTPUT_DIR) and os.path.exists(TOKENIZER_FILE):
        print(f"\nFound existing model in '{OUTPUT_DIR}'. Loading saved model and tokenizer...")
        tokenizer = A1Tokenizer.from_file(TOKENIZER_FILE)
        model = A1RNNModel.from_pretrained(OUTPUT_DIR)
    else:
        print("\nNo saved model found. Initiating training process...")
        
        # 1. Build Tokenizer
        print("Building vocabulary and tokenizer...")
        tokenizer = build_tokenizer(TRAIN_FILE, max_voc_size=10000, model_max_length=64)
        tokenizer.save(TOKENIZER_FILE)
        
        # 2. Initialize Model
        print("Initializing A1RNNModel...")
        config = A1RNNModelConfig(vocab_size=len(tokenizer), embedding_size=256, hidden_size=512)
        model = A1RNNModel(config)
        
        # 3. Setup Training Arguments the Proper Way
        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            optim='adamw_torch_fused',
            eval_strategy='epoch',
            use_cpu=not torch.cuda.is_available() and not torch.mps.is_available(),
            learning_rate=1e-2,
            num_train_epochs=10,
            per_device_train_batch_size=512,
            per_device_eval_batch_size=512,
            dataloader_num_workers=4,
            bf16=True, 
            tf32=True,
        )

        # 4. Setup and Run Trainer
        trainer = A1Trainer(
            model=model, 
            args=args, 
            train_dataset=dataset['train'], 
            eval_dataset=dataset['val'], 
            tokenizer=tokenizer
        )
        
        print("Starting training loop...")
        trainer.train()

    # --- Analysis Pipeline (Part 5) ---
    print("\n" + "="*40)
    print("--- Running Evaluation and Analysis ---")
    print("="*40)
    
    # Task 5.1: Next Word Prediction
    prompt_text = "she lives in san"
    print(f"\n[Task 5.1] Predicting next words for: '{prompt_text}'")
    predictions = predict_next_word(model, tokenizer, prompt_text, k=5, device=device)
    for word, score in predictions:
        print(f"  - {word}: {score:.4f}")
        
    # Task 5.2: Perplexity Evaluation
    print("\n[Task 5.2] Computing Validation Perplexity (this may take a minute)...")
    val_perp = compute_perplexity(model, dataset['val'], tokenizer, batch_size=128, device=device)
    print(f"  -> Validation Perplexity: {val_perp:.2f}")
    
    # Task 5.3: Word Embedding Geometry Inspection
    target_word = "sweden"
    print(f"\n[Task 5.3] Finding Nearest Neighbors for '{target_word}'...")
    neighbors = nearest_neighbors(model.embedding, tokenizer.str_to_int, tokenizer.int_to_str, target_word, n_neighbors=5)
    if neighbors:
        for word, score in neighbors:
            print(f"  - {word}: {score:.4f}")
    else:
        print(f"  -> '{target_word}' not found in the vocabulary. Try a more common word.")