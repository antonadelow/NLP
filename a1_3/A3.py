from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer
import evaluate
from transformers import Trainer
from transformers.trainer_callback import ProgressCallback
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
import json
import os
import time
import torch
import torch.nn as nn

def format_input_output(example):
    # `messages` is a list of messages, each with a `content` string and a `role`.
    messages = example['messages']

    # TODO: implement this
    prompt = ""
    response = ""
    
    for msg in messages:
        if msg['role'] == 'system':
            prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg['role'] == 'user':
            prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg['role'] == 'assistant':
            # The assistant response is the target we want the model to generate
            response = f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"

    return {"prompt": prompt, "response": response}

def tokenize_helper(example):
    prompt = example['prompt']     # Created in the previous step
    response = example['response'] # Created in the previous step

    # TODO: your work goes here.

    prompt_tokens = tokenizer(prompt)
    response_tokens = tokenizer(response)

    input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"]
    attention_mask = prompt_tokens["attention_mask"] + response_tokens["attention_mask"]
    
    # -100 tells PyTorch's CrossEntropyLoss to ignore these tokens
    labels = [-100] * len(prompt_tokens["input_ids"]) + response_tokens["input_ids"]

    return {
        "input_ids": input_ids,       # Input token ids of the prompt and response
        "attention_mask": attention_mask,  # Attention mask of the prompt and response
        "labels": labels,          # Output token ids of the prompt (masked) and response
    }

def data_collator(batch):
    """
    Create a custom collate function for causal language modeling.

    Args:
        batch: List of examples, each with 'input_ids', 'attention_mask', 'labels'
        tokenizer: Tokenizer with pad_token_id
    """

    input_ids_list = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
    attention_masks_list = [torch.tensor(example["attention_mask"], dtype=torch.long) for example in batch]
    labels_list = [torch.tensor(example['labels'], dtype=torch.long) for example in batch]

    # Find max length in this batch
    max_len = max(x.size(0) for x in input_ids_list)

    # Helper pad function
    def pad_to_max(x_list, pad_value):
        padded = []
        for x in x_list:
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                pad_tensor = torch.full((pad_len,), pad_value, dtype=x.dtype)
                x = torch.cat([x, pad_tensor], dim=0)
            padded.append(x)
        return torch.stack(padded, dim=0)

    # Use tokenizer.pad_token_id for inputs, 0 for attention_mask, -100 for labels
    pad_id = tokenizer.pad_token_id

    batch_input_ids = pad_to_max(input_ids_list, pad_value=pad_id)
    batch_attention_mask = pad_to_max(attention_masks_list, pad_value=0)
    batch_labels = pad_to_max(labels_list, pad_value=-100)

    batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
    return batch

class RougeMetricComputer:
    """
    Stateful metric for batch_eval_metrics=True.

    It:
      - accumulates predictions and references across batches
      - computes ROUGE-L once at the end (compute_result=True)
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = evaluate.load("rouge")
        self.all_predictions = []
        self.all_references = []

    def __call__(self, eval_pred, compute_result=False):
        """Accumulate predictions and compute at the end."""
        logits, labels = eval_pred
        
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        pred_ids = shift_logits.argmax(axis=-1)
        
        # Collect decoded answer-span text from each example in the batch
        for p, lbl in zip(pred_ids, shift_labels):
            mask = lbl != -100
            if mask.sum() == 0:
                continue

            ref_ids = lbl[mask]
            pred_ids_filtered = p[mask]

            eos_id = self.tokenizer.vocab.get('<|im_end|>', self.tokenizer.eos_token_id)

            ref_text = self.tokenizer.decode(ref_ids)
            pred_text = self.tokenizer.decode(
                pred_ids_filtered,
                eos_token_id=eos_id
            )

            self.all_references.append(ref_text.strip())
            self.all_predictions.append(pred_text.strip())

        # Only compute at the very end of eval
        if compute_result:
            if len(self.all_references) > 0:
                scores = self.rouge.compute(
                    predictions=self.all_predictions,
                    references=self.all_references,
                )

                # Clear accumulated data for next eval call
                self.all_predictions = []
                self.all_references = []
                return {"rougeL": scores["rougeL"]}
            else:
                return {}
        else:
            return {}

def make_trainer(model, training_args, tokenized_dataset=None):
    train_ds = tokenized_dataset["train"] if tokenized_dataset is not None else tokenized_ds_sft["train"]
    eval_ds = tokenized_dataset["test"] if tokenized_dataset is not None else tokenized_ds_sft["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if type(cb).__name__ != "NotebookProgressCallback"
    ]
    trainer.add_callback(ProgressCallback)
    return trainer

def num_trainable_parameters(model):
    """Count number of trainable parameters.

    Args:
        model: A PyTorch module.
    """
    # TODO: Add your code here
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def replace_layers(model, named_layers):
    """
    Replace submodules in `model` by name.
    """
    for name, layer in named_layers.items():
        components = name.split(".")
        submodule = model
        for comp in components[:-1]:
            submodule = getattr(submodule, comp)
        setattr(submodule, components[-1], layer)
    return model


def extract_lora_targets(model):
    # TODO: Add your code here
    targets = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            targets[name] = module
    return targets

def has_saved_baseline_model(output_dir):
    if not os.path.isdir(output_dir):
        return False

    config_path = os.path.join(output_dir, "config.json")
    model_bin = os.path.join(output_dir, "pytorch_model.bin")
    model_safe = os.path.join(output_dir, "model.safetensors")

    return os.path.isfile(config_path) and (os.path.isfile(model_bin) or os.path.isfile(model_safe))

def has_saved_lora_model(output_dir):
    if not os.path.isdir(output_dir):
        return False

    return os.path.isfile(os.path.join(output_dir, "lora_state.pt"))

def build_lora_model(model_name_or_path, r=8, alpha=16):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    for param in model.parameters():
        param.requires_grad = False

    lora_targets = extract_lora_targets(model)
    lora_wrapped_layers = {
        name: LoRALayer(layer, r, alpha).to("cuda" if torch.cuda.is_available() else "cpu")
        for name, layer in lora_targets.items()
    }

    replace_layers(model, lora_wrapped_layers)
    return model

class LoRALayer(nn.Module):
    def __init__(self, W, r, alpha):
        super().__init__()
        # TODO: Add your code here

        self.W = W
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = W.in_features
        out_features = W.out_features

        # Create low-rank matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # Freeze the original layer's parameters
        self.W.weight.requires_grad = False
        if self.W.bias is not None:
            self.W.bias.requires_grad = False

        # Initialize A with normal distribution and B with zeros
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # TODO: Add your code here
        original_output = self.W(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return original_output + lora_output


def test_model_generation(model, tokenizer, user_query, system_msg="You are a helpful assistant."):
        """
        Formats a query using Task 1.2 ChatML format, runs generation, 
        and extracts only the new tokens produced by the assistant.
        """
        # 1. Replicate Task 1.2 Prompt Formatting Exactly
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n" # Trigger token sequence for generation

        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 2. Configure safe generation arguments
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 3. Slice out the prompt tokens so we ONLY decode what the assistant generated
        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = output_ids[0][prompt_length:]
        
        return tokenizer.decode(generated_tokens).strip()


if __name__ == "__main__":

    SEED = 101
    MAX_TRAIN_SAMPLES = 5000
    MAX_TEST_SAMPLES = 400
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

    model_name_or_path = MODEL_NAME

    smoltalk = load_dataset("HuggingFaceTB/smoltalk", 'all')

    smoltalk_simplified = smoltalk.filter(lambda row: len(row['messages']) <= 3 and all(len(m['content']) <= 256 for m in row['messages']))
    smoltalk_simplified = DatasetDict({
        "train": smoltalk_simplified["train"].select(range(MAX_TRAIN_SAMPLES)),
        "test": smoltalk_simplified["test"].select(range(MAX_TEST_SAMPLES)),
    })

    ds_sft = smoltalk_simplified.map(format_input_output, load_from_cache_file=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    compute_metrics = RougeMetricComputer(tokenizer)

    print("\n" + "=" * 80)
    print("EVALUATING PRETRAINED MODEL")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

    pretrained_model.resize_token_embeddings(len(tokenizer))

    pretrained_eval_args = TrainingArguments(
        eval_strategy="no",
        per_device_eval_batch_size=1,
        bf16=True, fp16=False, # This may need to be changed, depending on the model you selected
        report_to="none",
        batch_eval_metrics=True,
        eval_accumulation_steps=1,
    )

    tokenized_ds_sft = ds_sft.map(tokenize_helper, load_from_cache_file=False)
    pretrained_trainer = make_trainer(pretrained_model, pretrained_eval_args, tokenized_dataset=tokenized_ds_sft)

    t0 = time.perf_counter()
    pretrained_eval_metrics = pretrained_trainer.evaluate()
    pretrained_eval_time = time.perf_counter() - t0

    pretrained_eval_loss = float(pretrained_eval_metrics["eval_loss"])
    pretrained_rougeL = pretrained_eval_metrics.get("eval_rougeL", None)

    print("\nPRETRAINED EVAL METRICS:")
    print(json.dumps(pretrained_eval_metrics, indent=2))

    print("\n" + "=" * 80)
    print("TRAINING BASELINE MODEL (SFT)")
    print("=" * 80)

    baseline_output_dir = "a1_3/baseline_model"
    baseline_training_args = TrainingArguments(
        output_dir=baseline_output_dir,
        eval_strategy="epoch",
        logging_steps=500,
        save_strategy="no",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        bf16=torch.cuda.is_available(), fp16=False,
        report_to="none",
        batch_eval_metrics=True,
        eval_accumulation_steps=1,
    )

    if has_saved_baseline_model(baseline_output_dir):
        print(f"Found saved baseline model in {baseline_output_dir}. Loading for eval.")
        base_model = AutoModelForCausalLM.from_pretrained(baseline_output_dir).to(device)
    else:
        print("No saved baseline model found. Training and saving.")
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        baseline_trainer = make_trainer(base_model, baseline_training_args, tokenized_dataset=tokenized_ds_sft)
        baseline_trainer.train()
        baseline_trainer.save_model(baseline_output_dir)

    print(f"Trainable parameters (Baseline): {num_trainable_parameters(base_model)}")
    baseline_trainer = make_trainer(base_model, baseline_training_args, tokenized_dataset=tokenized_ds_sft)
    baseline_eval_metrics = baseline_trainer.evaluate()
    print("\nBASELINE (SFT) EVAL METRICS:")
    print(json.dumps(baseline_eval_metrics, indent=2))

    # --- EDITS MADE HERE: Adding the final LoRA Task logic ---
    print("\n" + "=" * 80)
    print("TRAINING LoRA MODEL")
    print("=" * 80)
    
    lora_output_dir = "a1_3/lora_model"
    r = 8
    alpha = 16

    if has_saved_lora_model(lora_output_dir):
        print(f"Found saved LoRA model in {lora_output_dir}. Loading for eval.")
        lora_model = build_lora_model(model_name_or_path, r=r, alpha=alpha)
        lora_state_path = os.path.join(lora_output_dir, "lora_state.pt")
        lora_model.load_state_dict(torch.load(lora_state_path, map_location="cpu"))
        lora_model = lora_model.to(device)
    else:
        print("No saved LoRA model found. Training and saving.")
        lora_model = build_lora_model(model_name_or_path, r=r, alpha=alpha)
    print(f"Trainable parameters (LoRA): {num_trainable_parameters(lora_model)}")
    lora_model.resize_token_embeddings(len(tokenizer))

    lora_training_args = TrainingArguments(
        output_dir=lora_output_dir,
        eval_strategy="epoch",
        logging_steps=500,
        save_strategy="no",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        bf16=torch.cuda.is_available(), fp16=False,
        report_to="none",
        batch_eval_metrics=True,
        eval_accumulation_steps=1,
    )

    lora_trainer = make_trainer(lora_model, lora_training_args, tokenized_dataset=tokenized_ds_sft)
    if not has_saved_lora_model(lora_output_dir):
        lora_trainer.train()
        os.makedirs(lora_output_dir, exist_ok=True)
        torch.save(lora_model.state_dict(), os.path.join(lora_output_dir, "lora_state.pt"))

    lora_eval_metrics = lora_trainer.evaluate()
    print("\nLoRA EVAL METRICS:")
    print(json.dumps(lora_eval_metrics, indent=2))

    test_queries = [
        "What is the capital of France?",
        "Give me a quick recipe for chocolate chip cookies.",
        "Write a short, three-sentence poem about coding in Python.",
        "Explain quantum computing like I am five years old."
    ]

    # Run the interactive comparison
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- TEST CASE {i}: '{query}' ---")
        
        print("\n[1. PRETRAINED BASE MODEL ANSWER]:")
        try:
            print(test_model_generation(pretrained_model, tokenizer, query))
        except Exception as e:
            print(f"Generation failed: {e}")
            
        print("\n[2. FULL SFT BASELINE MODEL ANSWER]:")
        try:
            print(test_model_generation(base_model, tokenizer, query))
        except Exception as e:
            print(f"Generation failed: {e}")
            
        print("\n[3. LoRA TRAINED MODEL ANSWER]:")
        try:
            print(test_model_generation(lora_model, tokenizer, query))
        except Exception as e:
            print(f"Generation failed: {e}")
        print("-" * 60)

#Task 2.2: Evaluating the pre-trained model
#Now, we have all the pieces to evaluate our baseline model that has not been instruction-tuned.
#The following code will compute the loss on the test set as well as the ROUGE-L score. You will later compare these scores to the models that you train.
#Why do you think the ROUGE-L score is as high as it is, even without any training for instruction-following?

#Because the evaluation dataset is made up of standard human text, and the ROUGE-L metric simply measures how well the model's 
#next-token predictions overlap with the actual words in the text, the model naturally scores high. 
#It does not need instruction tuning to know which English word logically follows a given sentence.

#Task 3.1: Training the full model
#Next, we train the pre-trained model using SFT over all the parameters, then calculate the metrics and outputs to evaluate how well it follows instructions.
#How do the results differ from those in the previous step?

#The evaluation loss dropped dramatically from 3.1639 to 1.2458, and the true evaluation ROUGE-L climbed to 0.6626.
#The pre-trained model was caught in token repetition loops (repeating variations of "What is the capital of Germany? ostering"). 
#The Full SFT model broke out of this pattern, correctly recognized the ChatML syntax structure (<|im_start|>assistant), 
#and successfully switched to an instruction-following role, producing direct answers like "The capital of France is Paris."

#Task 4.4: Qualitative inspection
#Run the three models interactively on some examples of your own choice (either taken from the training or test sets, 
#or created by yourself). 
#Do your models seem to have learned the instruction-following behavior (at least to some extent)? 
#Do they respond to user queries sensibly?

#The pre-trained base model fails at instruction following because it has no concept of the chat boundary template, 
#causing it to treat user inputs as text documents that should be completed, and collapse into infinite repetition loops 
#or irrelevant web vocabulary. Both the full SFT baseline and the LoRA-trained models successfully internalize the 
#ChatML formatting structure and adopt the conversational persona of an assistant, although the restricted capacity 
#of the 135-million parameter architecture causes the LoRA variant to occasionally bleed trailing system vocabulary 
#at its termination boundaries. While both fine-tuned models demonstrate reliability for straightforward factual retrieval tasks, 
#their creative and explanatory depth degrades into overly simplistic or generic placeholders when dealing with more 
#complex prompts, and quality remains constrained by the small model size and the training limit.
