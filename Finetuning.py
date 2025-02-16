import os
from dotenv import load_dotenv
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

from src.helper import (
    validate_sql,
    preprocess_query,
    clean_sql_output,
    log_training_metrics,
    timestamp,
    get_logger,
    format_mistral_prompt
)

logger = get_logger(__name__)

class SQLModelTrainer:
    """Optimized for Mistral-7B SQL Generation (training mode)"""

    def __init__(self):
        load_dotenv()
        
        # Configuration
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.output_dir = f"mistral_sql_{timestamp()}"
        self.max_length = 512
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_model()

    def _setup_model(self):
        """Mistral-specific setup with 4-bit quantization for training (entire model on GPU)"""
        try:
            # 4-bit Quantization Config.
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=False  # Offloading disabled for training.
            )
            
            # Tokenizer setup
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                padding_side="left",
                truncation_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Use current GPU device so that the model is loaded on the same device as training.
            device_map = {"": torch.cuda.current_device()}
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                quantization_config=self.quant_config,
                device_map=device_map,
                torch_dtype=torch.float16
                # Note: offload_folder is omitted since no offloading is desired.
            )
            
            # Enable gradient checkpointing for memory efficiency.
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Memory-efficient LoRA Config for fine-tuning.
            self.lora_config = LoraConfig(
                r=8,             # Reduced to save memory.
                lora_alpha=16,   # Reduced for memory efficiency.
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                inference_mode=False
            )
            
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()
            logger.info("Mistral model loaded successfully for training (4-bit quantization, loaded on current GPU)")

        except Exception as e:
            logger.error(f"Model setup failed: {str(e)}")
            raise

    def prepare_training_data(self, data_path: str):
        """Mistral-optimized data processing"""
        try:
            dataset = load_dataset('csv', data_files={'train': data_path})
            dataset = dataset['train'].train_test_split(test_size=0.1)

            def format_example(example):
                if not validate_sql(example['sql_query'])[0]:
                    return None
                
                processed = format_mistral_prompt(
                    preprocess_query(example['natural_language']),
                    clean_sql_output(example['sql_query'])
                )
                
                tokenized = self.tokenizer(
                    processed,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None
                )
                tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized

            processed_dataset = {}
            for split in ['train', 'test']:
                processed_examples = []
                for example in dataset[split]:
                    result = format_example(example)
                    if result is not None:
                        processed_examples.append(result)
                
                if not processed_examples:
                    raise ValueError(f"No valid examples found in the {split} set.")
                
                processed_dataset[split] = {
                    key: [ex[key] for ex in processed_examples]
                    for key in processed_examples[0].keys()
                }
                processed_dataset[split] = dataset[split].from_dict(processed_dataset[split])
                logger.info(f"Processed {len(processed_dataset[split])} examples for {split} set")

            return processed_dataset

        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise

    def execute_training(self, train_data, val_data):
        """Training process with memory optimizations and reduced batch size"""
        try:
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=1,  # Reduced batch size for memory constraints.
                per_device_eval_batch_size=1,
                num_train_epochs=10,
                learning_rate=1e-4,
                gradient_accumulation_steps=8,  # Simulate a larger effective batch size.
                fp16=True,
                optim="paged_adamw_8bit",
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=50,
                save_strategy="steps",
                save_steps=50,
                report_to="tensorboard",
                remove_unused_columns=False,
                gradient_checkpointing=True,
                max_grad_norm=0.3,
                warmup_ratio=0.03
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            )

            logger.info(f"Training started at {timestamp()}")
            trainer.train()
            
            save_path = os.path.join(self.output_dir, "final_model")
            self.model.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    try:
        trainer = SQLModelTrainer()
        dataset = trainer.prepare_training_data("nl_sql_dataset.csv")
        trainer.execute_training(dataset['train'], dataset['test'])
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
