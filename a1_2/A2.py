
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutput
from a1_1.A1 import A1Tokenizer, A1Trainer, predict_next_word, compute_perplexity, build_tokenizer
import torch.nn.functional as F
from safetensors.torch import load_file
import os

class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers


class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        # TODO: initalize components here
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_states):
        # SwiGLU: (SiLU(xW_gate) ⊗ xW_up)W_down
        gate = self.act(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)

class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        # TODO: set up W_q, W_k, W_v, W_o here
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # TODO: set up normalizers here
        self.q_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, rope_rotations):
        b, m, _ = hidden_states.size()
        
        # Compute and normalize Q and K
        q = self.q_norm(self.W_q(hidden_states))
        k = self.k_norm(self.W_k(hidden_states))
        v = self.W_v(hidden_states)
        
        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, rope_rotations)
        
        # Scaled dot-product attention (is_causal=True applies the lower-triangular mask automatically)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape back to (batch, seq_len, hidden_size)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(b, m, self.hidden_size)
        
        return self.W_o(attn_out)


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        # TODO: set up attention, MLP, and normalizers here.
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = A2Attention(config)
        
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = A2MLP(config)

    def forward(self, hidden_states, rope_rotations):
        # Pre-Norm Architecture with Residual Connections
        h = hidden_states + self.attention(self.attn_norm(hidden_states), rope_rotations)
        out = h + self.mlp(self.mlp_norm(h))
        return out


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # TODO: put all transformer decoder layers in a ModuleList.
        self.layers = nn.ModuleList(
            [A2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, input_ids, labels=None):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers

        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, rope_rotations)
        x = self.norm(x)
        logits = self.unembedding(x)
        
        # TODO: Compute the loss as in Assignment 1 if labels is not None.
        loss = None
        if labels is not None:
            # Shift the inputs by one position as we did in Assignment 1 
            # so that token t predicts token t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_func(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutput(logits=logits, loss=loss)

#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, topk=5, device='cpu'):
    model.eval()
    model.to(device)
    encodings = tokenizer([prompt], truncation=True, padding=False, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    generated_ids = input_ids.tolist()[0]
    
    # === CRITICAL FIX ===
    # Strip the trailing <EOS> token appended by the tokenizer so the model can continue the prompt
    if len(generated_ids) > 0 and generated_ids[-1] == getattr(tokenizer, 'eos_token_id', -1):
        generated_ids.pop()
    # ====================
    
    with torch.no_grad():
        for _ in range(max_length):
            current_input = torch.tensor([generated_ids]).to(device)
            current_input = current_input[:, -model.config.max_position_embeddings:]
            outputs = model(current_input)
            
            next_token_logits = outputs.logits[0, -1, :] / temperature
            top_logits, top_indices = torch.topk(next_token_logits, topk)
            probs = F.softmax(top_logits, dim=-1)
            
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_indices[next_token_idx].item()
            
            generated_ids.append(next_token)
            
            if next_token == getattr(tokenizer, 'eos_token_id', -1):
                break
                
    return " ".join([tokenizer.int_to_str.get(idx, "") for idx in generated_ids])
    
def compare_with_pretrained(prompt, max_length=50, device='cpu'):
    model_name = 'allenai/OLMo-2-0425-1B'
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = hf_model.generate(**inputs, max_new_tokens=max_length, temperature=0.8, top_k=5, do_sample=True)
    
    return hf_tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import TrainingArguments
    import os

    # --- Configuration ---
    TRAIN_FILE = 'a1_1/train.txt'  
    VAL_FILE = 'a1_1/val.txt'      
    OUTPUT_DIR = 'a1_2/trainer_output'
    TOKENIZER_FILE = 'a1_2/a2_tokenizer.pkl'
    
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
        
        # 1. Use HF only to read the text config file, then instantiate a raw model
        config = A2ModelConfig.from_pretrained(OUTPUT_DIR)
        model = A2Transformer(config)
        
        # 2. Locate the raw weight files saved by the Trainer
        safetensors_path = os.path.join(OUTPUT_DIR, "model.safetensors")
        bin_path = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
        
        # 3. Load the raw tensor dictionary directly into CPU memory
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No weight files found in {OUTPUT_DIR}")
            
        # 4. Force a strict, literal 1:1 weight injection. 
        # If a single parameter name or shape doesn't match perfectly, this will hard-crash.
        model.load_state_dict(state_dict, strict=True)
        print("--> State dict loaded with 100% strict identity.")
        
        # 5. Move the fully loaded model to your cluster GPU
        model = model.to(device)    
    else:
        print("\nNo saved model found. Initiating training process...")
        
        # 1. Build Tokenizer
        print("Building vocabulary and tokenizer...")
        tokenizer = build_tokenizer(TRAIN_FILE, max_voc_size=10000, model_max_length=64)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tokenizer.save(TOKENIZER_FILE)
        
        # 2. Initialize Model
        print("Initializing model...")
        config = A2ModelConfig(vocab_size=len(tokenizer), hidden_size=256, intermediate_size=256, num_attention_heads=4, num_hidden_layers=3, rope_theta=10000.0, max_position_embeddings=64, rms_norm_eps=1e-5)
        model = A2Transformer(config)
        
        # 3. Setup Training Arguments the Proper Way
        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            optim='adamw_torch_fused',
            eval_strategy='epoch',
            use_cpu=not torch.cuda.is_available() and not torch.mps.is_available(),
            learning_rate=1e-3,
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
    
    print("\n[Task 2.1] Computing Validation Perplexity (this may take a minute)...")
    val_perp = compute_perplexity(model, dataset['val'], tokenizer, batch_size=128, device=device)
    print(f"  -> Validation Perplexity: {val_perp:.2f}")

    # Task 3.1: Next Word Prediction
    prompt_text = "she lives in san"
    print(f"\n[Task 3.1] Predicting next words for: '{prompt_text}'")
    predictions = predict_next_word(model, tokenizer, prompt_text, k=5, device=device)
    for word, score in predictions:
        print(f"  - {word}: {score:.4f}")

    prompts_3_2 = [
        'In natural language processing, a Transformer',
        'Is Stockholm the capital of Sweden? Answer yes or no. The answer is',
        'Write a Python program that reverses a list.'
    ]
    
    for prompt in prompts_3_2:
        generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8, topk=5, device=device)
        print(f"Custom Model -> {generated}\n")
        
    for prompt in prompts_3_2:
        hf_generated = compare_with_pretrained(prompt, max_length=50, device=device)
        print(f"OLMo-2 -> {hf_generated}\n")