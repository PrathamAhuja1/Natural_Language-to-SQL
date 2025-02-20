import os
import torch
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from src.helper import (
    validate_sql,
    preprocess_query,
    clean_sql_output,
    timestamp,
    get_logger,
    format_t5_prompt
)

from dotenv import load_dotenv

# Ensure compatibility with multiprocessing
if __name__ == "__main__":
    logger = get_logger(__name__)

    class SQLModelTrainer:
        """Fine-tuning T5-Large for Text-to-SQL conversion with LoRA for improved accuracy."""

        def __init__(self):
            load_dotenv()
            self.model_name = "t5-large"
            self.output_dir = f"t5_sql_{timestamp()}"
            self.max_length = 256
            self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
            os.makedirs(self.output_dir, exist_ok=True)

            # Prefer BF16 if supported, otherwise FP16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16

            self._setup_model()

        def _setup_model(self):
            """Set up T5-Large with LoRA for fine-tuning."""
            try:
                # Initialize the tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_auth_token=self.hf_token
                )

                # Load T5-Large model on the current GPU
                device_map = {"": torch.cuda.current_device()}
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    use_auth_token=self.hf_token,
                    torch_dtype=self.torch_dtype,
                    device_map=device_map
                )

                # Enable gradient checkpointing to reduce memory footprint
                self.model.gradient_checkpointing_enable()

                # Set up LoRA for efficient fine-tuning
                self.lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q", "v"],
                    lora_dropout=0.05,
                    task_type="SEQ_2_SEQ_LM",
                    inference_mode=False
                )
                self.model = get_peft_model(self.model, self.lora_config)
                self.model.print_trainable_parameters()

                logger.info("T5-Large model loaded successfully with LoRA for fine-tuning.")
            except Exception as e:
                logger.error(f"Model setup failed: {str(e)}")
                raise

        def prepare_training_data(self, data_path: str):
            """Prepare training data for T5-Large text-to-SQL fine-tuning."""
            try:
                dataset = load_dataset('csv', data_files={'train': data_path})
                dataset = dataset['train'].train_test_split(test_size=0.1)

                def format_example(example):
                    # Validate SQL structure
                    valid, _ = validate_sql(example['sql_query'])
                    if not valid:
                        return None
                    # Preprocess natural language and SQL query
                    nl = preprocess_query(example['natural_language'])
                    sql = clean_sql_output(example['sql_query'])
                    # For T5, the input prompt is "translate English to SQL: <natural language>"
                    prompt = format_t5_prompt(nl)
                    # Tokenize the input prompt
                    input_encodings = self.tokenizer(
                        prompt,
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    # Tokenize the target SQL query
                    with self.tokenizer.as_target_tokenizer():
                        target_encodings = self.tokenizer(
                            sql,
                            max_length=self.max_length,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt"
                        )
                    features = {
                        "input_ids": input_encodings["input_ids"].squeeze(),
                        "attention_mask": input_encodings["attention_mask"].squeeze(),
                        "labels": target_encodings["input_ids"].squeeze()
                    }
                    return features

                processed_dataset = {}
                for split in ['train', 'test']:
                    processed_examples = []
                    for example in dataset[split]:
                        result = format_example(example)
                        if result is not None:
                            processed_examples.append(result)
                    if not processed_examples:
                        raise ValueError(f"No valid examples found in the {split} set.")
                    # Convert list of dictionaries to a dataset dict
                    processed_dataset[split] = {key: [ex[key] for ex in processed_examples]
                                                 for key in processed_examples[0].keys()}
                    processed_dataset[split] = dataset[split].from_dict(processed_dataset[split])
                    logger.info(f"Processed {len(processed_dataset[split])} examples for {split} set.")
                return processed_dataset
            except Exception as e:
                logger.error(f"Data processing failed: {str(e)}")
                raise

        def execute_training(self, train_data, val_data):
            """Execute fine-tuning of T5-Large model."""
            try:
                torch.backends.cudnn.benchmark = True
                training_args = TrainingArguments(
                    output_dir=self.output_dir,
                    per_device_train_batch_size=4,  # Increase if memory allows
                    per_device_eval_batch_size=4,
                    num_train_epochs=5,  # Ensure the model has enough epochs to converge
                    learning_rate=5e-5,  # Experiment with different learning rates
                    gradient_accumulation_steps=4,  # Adjust based on memory constraints
                    bf16=(self.torch_dtype == torch.bfloat16),
                    fp16=(self.torch_dtype == torch.float16),
                    optim="adamw_torch",
                    logging_steps=200,  # Reduce logging frequency
                    evaluation_strategy="steps",
                    eval_steps=1000,  # Evaluate less frequently
                    save_strategy="steps",
                    save_steps=1000,  # Save less frequently
                    report_to="tensorboard",
                    remove_unused_columns=False,
                    gradient_checkpointing=True,
                    max_grad_norm=1.0,  # Experiment with different gradient clipping values
                    warmup_ratio=0.1,  # Increase warmup ratio for better convergence
                    dataloader_num_workers=0,  # Temporarily set to 0 to debug
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    lr_scheduler_type="linear"  # Use a learning rate scheduler
                )
                data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_data,
                    eval_dataset=val_data,
                    data_collator=data_collator
                )
                logger.info(f"Training started at {timestamp()}")
                trainer.train()
                save_path =  "final_model"
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
