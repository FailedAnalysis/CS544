class Generator:

    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        logger.info(f"Loading model and tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
             self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.doc_dict = {}
        self.qa_df = None
        self.tokenized_dataset = None

    def build_doc_dict(self, documents_df):
        logger.info("Building document dictionary from DataFrame...")
        documents_df["id"] = documents_df["id"].astype(str)
        documents_df["passage"] = documents_df["passage"].fillna("").astype(str)
        self.doc_dict = dict(zip(documents_df["id"], documents_df["passage"]))
        logger.info(f"Built doc_dict with {len(self.doc_dict)} documents.")

    def create_prompt(self, question, context, answer=None):
        if answer is not None:
            prompt = f"Question: {question}\nContext: {context}\nAnswer: {answer}"
        else:
            prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        return prompt

    def build_truncated_prompt(self, question, context, max_new_tokens):
        question = str(question)
        context = str(context)

        max_positions = self.model.config.max_position_embeddings
        allowed_prompt_length = max_positions - max_new_tokens

        q_part_template = f"Question: {question}\nContext: "
        a_part_template = "\nAnswer:"

        q_tokens_len = len(self.tokenizer.encode(q_part_template, add_special_tokens=False))
        a_tokens_len = len(self.tokenizer.encode(a_part_template, add_special_tokens=False))

        fixed_length = q_tokens_len + a_tokens_len

        allowed_for_context = max(0, allowed_prompt_length - fixed_length)

        if not isinstance(context, str) or not context.strip():
             truncated_context = ""
        else:
             context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
             truncated_context_tokens = context_tokens[:allowed_for_context]
             truncated_context = self.tokenizer.decode(truncated_context_tokens, skip_special_tokens=True)

        truncated_prompt = f"Question: {question}\nContext: {truncated_context}\nAnswer:"

        final_prompt_length = len(self.tokenizer.encode(truncated_prompt, add_special_tokens=True))

        if final_prompt_length > allowed_prompt_length:
             logger.warning(
                 f"Final truncated prompt length ({final_prompt_length}) still exceeds allowed ({allowed_prompt_length}) "
                 f"before adding generation tokens. Review truncation logic or parameters."
             )

        full_raw_prompt_length = len(self.tokenizer.encode(f"Question: {question}\nContext: {context}\nAnswer:", add_special_tokens=True))
        if final_prompt_length < full_raw_prompt_length:
             logger.debug(
                 f"Prompt truncated from {full_raw_prompt_length} tokens to {final_prompt_length} tokens "
                 f"(allowed prompt length {allowed_prompt_length}). Query: {question[:50]}..."
             )

        return truncated_prompt


    def prepare_training_data(self, train_df):
        if not self.doc_dict:
            raise ValueError("Document dictionary is empty. Call build_doc_dict(documents_df) first.")
        if train_df.empty:
             logger.warning("Input training DataFrame is empty. Skipping data preparation.")
             self.qa_df = pd.DataFrame(columns=["question", "relevant_docs", "answer", "prompt"])
             return


        logger.info(f"Preparing training data from DataFrame ({len(train_df)} rows)...")
        records = []
        train_df['relevant_passage_ids'] = train_df['relevant_passage_ids'].apply(
             lambda x: [str(i).strip() for i in (eval(str(x)) if isinstance(x, str) else x) if str(i).strip()] if pd.notna(x) else []
        )
        train_df['question'] = train_df['question'].fillna("").astype(str)
        train_df['answer'] = train_df['answer'].fillna("").astype(str)


        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Preparing training data"):
            question = row["question"]
            answer = row["answer"]
            doc_ids = row["relevant_passage_ids"]

            relevant_texts = []
            for pid in doc_ids:
                if pid in self.doc_dict:
                    relevant_texts.append(self.doc_dict[pid])
                else:
                    logger.warning(f"Relevant passage ID {pid} not found in doc_dict for question ID {row.get('id', 'N/A')}. Skipping passage for training prompt.")

            combined_passages = " ".join(relevant_texts)

            prompt = self.create_prompt(question, combined_passages, answer)

            records.append({
                "question": question,
                "relevant_docs": combined_passages,
                "answer": answer,
                "prompt": prompt
            })

        self.qa_df = pd.DataFrame(records)
        logger.info(f"Prepared {len(self.qa_df)} training records.")

    def tokenize_training_data(self, max_length=512):
        if self.qa_df is None or self.qa_df.empty:
            raise ValueError("Training data not prepared or is empty. Cannot tokenize.")

        logger.info("Converting prepared data to Dataset and tokenizing...")
        dataset = Dataset.from_pandas(self.qa_df[["prompt"]])

        model_max_length = self.model.config.max_position_embeddings
        if max_length > model_max_length:
            logger.warning(f"Requested max_length ({max_length}) exceeds model max_position_embeddings ({model_max_length}). Using model_max_length.")
            max_length = model_max_length
        if max_length <= 0:
             logger.error("max_length must be positive. Tokenization failed.")
             self.tokenized_dataset = None
             return


        def tokenize_function(examples):
            tokenized_inputs = self.tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            return tokenized_inputs

        self.tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt"])
        logger.info(f"Tokenization complete. Using max_length={max_length}.")


    def train_model(self, output_dir="output/generator-finetuned", num_train_epochs=1, batch_size=4,
                    gradient_accumulation_steps=8, logging_steps=50, learning_rate=2e-5):
        if self.tokenized_dataset is None or len(self.tokenized_dataset) == 0:
            raise ValueError("Tokenized training data not found or is empty. Cannot train.")

        logger.info("Setting up PEFT/LoRA for training...")

        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        try:
             peft_model = get_peft_model(self.model, lora_config)
             peft_model.print_trainable_parameters()
             logger.info("PEFT model prepared successfully.")
        except Exception as e:
             logger.error(f"Error applying PEFT config: {e}")
             logger.error("Please check target_modules for your specific Llama3 model.")
             raise


        logger.info("Setting up training arguments and Trainer...")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim="paged_adamw_8bit",
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            save_steps=logging_steps * 5,
            save_total_limit=3,
            evaluation_strategy="no",
            logging_dir=f"{output_dir}/logs",
            report_to="tensorboard",
            run_name=f"llama3_{output_dir}_run",
            push_to_hub=False,
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Generator training complete!")

        peft_model.save_pretrained(output_dir)
        logger.info(f"LoRA adapters saved to {output_dir}")


    def evaluate_generator(self, eval_df, max_eval_samples=100, max_new_tokens=64,
                           do_sample=False, seed=42):
        if not self.doc_dict:
            raise ValueError("Document dictionary is empty. Ensure build_doc_dict(documents_df) has been called.")
        if eval_df.empty or 'relevant_passage_ids' not in eval_df.columns or 'answer' not in eval_df.columns:
             logger.warning("Evaluation DataFrame is empty or missing required columns ('relevant_passage_ids', 'answer'). Skipping evaluation.")
             return {}

        logger.info(f"Evaluating generator model on {min(len(eval_df), max_eval_samples)} samples...")

        subset_df = eval_df.sample(n=min(len(eval_df), max_eval_samples), random_state=seed).reset_index(drop=True)

        predictions = []
        references = []

        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Generation device: {device}")
        self.model.to(device)

        gen_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        try:
            bleu_metric = evaluate.load("bleu")
            rouge_metric = evaluate.load("rouge")
            logger.info("Evaluation metrics loaded.")
        except Exception as e:
            logger.error(f"Failed to load evaluation metrics: {e}. Evaluation will skip metric calculation.")
            bleu_metric = None
            rouge_metric = None

        with torch.no_grad():
            for i, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Generating for Evaluation"):
                question = str(row["question"])
                gold_answer = str(row["answer"])

                relevant_ids_raw = row["relevant_passage_ids"]
                doc_ids = []
                if isinstance(relevant_ids_raw, list):
                     doc_ids = [str(pid).strip() for pid in relevant_ids_raw if str(pid).strip()]
                elif isinstance(relevant_ids_raw, str):
                      try: doc_ids = [str(pid).strip() for pid in eval(relevant_ids_raw) if str(pid).strip()]
                      except (SyntaxError, NameError, TypeError): doc_ids = [pid.strip() for pid in relevant_ids_raw.strip().split(",") if pid.strip()]
                doc_ids = [d for d in doc_ids if d]

                relevant_texts = []
                for pid in doc_ids:
                    if pid in self.doc_dict:
                        relevant_texts.append(self.doc_dict[pid])
                    else:
                        logger.warning(f"Eval passage ID {pid} not found in doc_dict for question ID {row.get('id', 'N/A')}. Skipping passage.")

                combined_context = " ".join(relevant_texts)

                prompt = self.build_truncated_prompt(question, combined_context, max_new_tokens)

                try:
                    gen_output = gen_pipe(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                        do_sample=do_sample,
                        temperature=0.7 if do_sample else 1.0,
                        top_k=50 if do_sample else None,
                        top_p=0.95 if do_sample else None,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_full_text=False
                    )
                    if gen_output and isinstance(gen_output, list) and len(gen_output) > 0 and 'generated_text' in gen_output[0]:
                         pred_answer = gen_output[0]['generated_text'].strip()
                    else:
                         pred_answer = ""
                         logger.warning(f"Pipeline returned unexpected output for prompt starting '{prompt[:50]}...'. Output: {gen_output}")

                except Exception as e:
                    logger.error(f"Error during generation for prompt starting '{prompt[:50]}...': {e}")
                    pred_answer = ""


                predictions.append(pred_answer)
                references.append(gold_answer)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        evaluation_results = {}
        if bleu_metric:
            references_formatted = [[ans] for ans in references]
            try:
                bleu_score = bleu_metric.compute(predictions=predictions, references=references_formatted)
                evaluation_results.update({f"bleu_{k}": v for k,v in bleu_score.items()})
                logger.info(f"BLEU score: {bleu_score.get('bleu', 'N/A'):.4f}")
            except Exception as e:
                 logger.error(f"Error computing BLEU: {e}")

        if rouge_metric:
             try:
                  rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
                  evaluation_results.update(rouge_scores)
                  logger.info(f"ROUGE scores: {rouge_scores}")
             except Exception as e:
                  logger.error(f"Error computing ROUGE: {e}")

        return evaluation_results, predictions, references


    def predict(self, test_df, max_new_tokens=64, do_sample=False, seed=42):
        if not self.doc_dict:
            raise ValueError("Document dictionary is empty. Ensure build_doc_dict(documents_df) has been called.")
        if test_df.empty or 'relevant_passage_ids' not in test_df.columns:
             logger.warning("Test DataFrame is empty or missing 'relevant_passage_ids'. Skipping prediction.")
             return test_df.copy().assign(predicted_answer=[""] * len(test_df))


        logger.info(f"Generating answers for test data ({len(test_df)} rows)...")

        results_df = test_df.copy()
        predictions = []

        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Generation device: {device}")
        self.model.to(device)

        gen_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        results_df['relevant_passage_ids'] = results_df['relevant_passage_ids'].apply(
             lambda x: [str(i).strip() for i in (eval(str(x)) if isinstance(x, str) else x) if str(i).strip()] if pd.notna(x) else []
        )
        results_df['question'] = results_df['question'].fillna("").astype(str)


        with torch.no_grad():
            for i, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Generating answers"):
                question = row["question"]
                doc_ids = row["relevant_passage_ids"]

                relevant_texts = []
                for pid in doc_ids:
                    if pid in self.doc_dict:
                        relevant_texts.append(self.doc_dict[pid])
                    else:
                        logger.warning(f"Predicted passage ID {pid} not found in doc_dict for question ID {row.get('id', 'N/A')}. Skipping passage.")

                combined_context = " ".join(relevant_texts)

                prompt = self.build_truncated_prompt(question, combined_context, max_new_tokens)

                try:
                    gen_output = gen_pipe(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                        do_sample=do_sample,
                        temperature=0.7 if do_sample else 1.0,
                        top_k=50 if do_sample else None,
                        top_p=0.95 if do_sample else None,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_full_text=False
                    )
                    if gen_output and isinstance(gen_output, list) and len(gen_output) > 0 and 'generated_text' in gen_output[0]:
                         pred_answer = gen_output[0]['generated_text'].strip()
                    else:
                         pred_answer = ""
                         logger.warning(f"Pipeline returned unexpected output for prompt starting '{prompt[:50]}...'. Output: {gen_output}")

                except Exception as e:
                    logger.error(f"Error during generation for prompt starting '{prompt[:50]}...': {e}")
                    pred_answer = ""

                predictions.append(pred_answer)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        results_df["predicted_answer"] = predictions
        logger.info("Answer generation complete.")
        return results_df


class RAGSystem:
    def __init__(self, retriever_model_path=None, generator_model_path=None,
                 retriever_base_model="all-mpnet-base-v2", reranker_base_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 generator_base_model="meta-llama/Meta-Llama-3-8B-Instruct"):
        logger.info("Initializing RAG System...")

        if retriever_model_path and os.path.exists(retriever_model_path):
             logger.info(f"Loading fine-tuned retriever model from {retriever_model_path}")
             self.retriever_model_instance = SentenceTransformer(retriever_model_path)
             logger.info("Loading base retriever model for RAGSystem...")
             base_retriever_model = SentenceTransformer(retriever_base_model)
             if retriever_model_path and os.path.exists(retriever_model_path):
                 try:
                     base_retriever_model.load(retriever_model_path)
                     logger.info(f"Loaded fine-tuned retriever weights from {retriever_model_path}")
                 except Exception as e:
                     logger.error(f"Failed to load fine-tuned retriever model from {retriever_model_path}: {e}")
                     logger.warning("Proceeding with base retriever model only.")
             self.retriever = Retriever(retriever_base_model, reranker_base_model)
             self.retriever.retriever_model = base_retriever_model

        else:
             logger.info(f"Loading base retriever model {retriever_base_model}")
             self.retriever = Retriever(retriever_base_model, reranker_base_model)


        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"Loading base generator model {generator_base_model} with quantization...")
        base_generator_model = AutoModelForCausalLM.from_pretrained(
            generator_base_model,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
        base_tokenizer = AutoTokenizer.from_pretrained(generator_base_model, trust_remote_code=True)
        if base_tokenizer.pad_token is None:
             base_tokenizer.pad_token = base_tokenizer.eos_token
        base_generator_model.config.pad_token_id = base_tokenizer.eos_token_id


        if generator_model_path and os.path.exists(generator_model_path):
             logger.info(f"Loading fine-tuned generator adapters from {generator_model_path}")
             try:
                 from peft import PeftModel
                 self.generator_model = PeftModel.from_pretrained(base_generator_model, generator_model_path)
                 logger.info("LoRA adapters loaded and merged with base model.")
             except Exception as e:
                 logger.error(f"Failed to load PEFT adapters from {generator_model_path}: {e}")
                 logger.warning("Proceeding with base generator model only.")
                 self.generator_model = base_generator_model
        else:
             logger.info("No fine-tuned generator path provided or path does not exist. Using base generator model.")
             self.generator_model = base_generator_model

        self.generator_tokenizer = base_tokenizer


        self.documents_df = None
        self.corpus_doc_dict = {}


    def load_corpus(self, documents_df):
        logger.info("Loading corpus into RAG system...")
        self.documents_df = documents_df.copy()
        self.retriever.load_corpus(self.documents_df)
        self.corpus_doc_dict = dict(zip(self.documents_df["id"].astype(str), self.documents_df["passage"].fillna("").astype(str)))
        logger.info("Corpus loaded into RAG system's internal storage.")


    def precompute_retriever_embeddings(self, batch_size=1024):
        if self.documents_df is None or self.documents_df.empty:
            raise ValueError("Corpus not loaded. Call load_corpus(documents_df) first.")
        self.retriever.precompute_corpus_embeddings(batch_size=batch_size)


    def answer_question(self, question: str, top_k_retrieval=5, initial_retrieval_k=100, use_reranking=True,
                        max_new_tokens=100, do_sample=False):
        if self.retriever.corpus_embeddings is None:
            logger.warning("Retriever embeddings not precomputed. Computing now...")
            try:
                self.precompute_retriever_embeddings()
            except Exception as e:
                logger.error(f"Failed to precompute embeddings: {e}. Cannot perform retrieval.")
                return "Error: Could not initialize retriever."

        if not self.corpus_doc_dict:
             logger.error("Corpus document dictionary not loaded. Cannot retrieve passage texts.")
             return "Error: Corpus not loaded."

        logger.info(f"Answering question: '{question}'")

        logger.info(f"Retrieving top {top_k_retrieval} passages...")
        retrieved_results = self.retriever.retrieve_top_k(
            question,
            top_k=top_k_retrieval,
            initial_retrieval_k=initial_retrieval_k,
            use_reranking=use_reranking
        )

        if not retrieved_results:
            logger.warning("No passages retrieved.")
            return "Could not find relevant information to answer the question."

        combined_context = " ".join([self.corpus_doc_dict.get(res.get('corpus_id', ''), '') for res in retrieved_results])

        if not combined_context.strip():
             logger.warning("Retrieved passages are empty or contain no text.")
             if not combined_context:
                 logger.warning("Combined context is empty.")


        logger.info(f"Combined context length: {len(combined_context)} characters.")

        logger.info("Generating answer...")

        model = self.generator_model
        tokenizer = self.generator_tokenizer
        max_positions = model.config.max_position_embeddings

        max_new_tokens_gen = max_new_tokens
        allowed_prompt_length = max_positions - max_new_tokens_gen

        q_part_template = f"Question: {question}\nContext: "
        a_part_template = "\nAnswer:"

        q_tokens_len = len(tokenizer.encode(q_part_template, add_special_tokens=False))
        a_tokens_len = len(tokenizer.encode(a_part_template, add_special_tokens=False))

        fixed_length = q_tokens_len + a_tokens_len
        allowed_for_context = max(0, allowed_prompt_length - fixed_length)

        context_to_truncate = combined_context

        if not isinstance(context_to_truncate, str) or not context_to_truncate.strip():
             truncated_context = ""
        else:
             context_tokens = tokenizer.encode(context_to_truncate, add_special_tokens=False)
             truncated_context_tokens = context_tokens[:allowed_for_context]
             truncated_context = tokenizer.decode(truncated_context_tokens, skip_special_tokens=True)

        prompt = f"Question: {question}\nContext: {truncated_context}\nAnswer:"

        final_prompt_length = len(tokenizer.encode(prompt, add_special_tokens=True))
        if final_prompt_length > allowed_prompt_length:
             logger.warning(f"Final prompt length ({final_prompt_length}) exceeds allowed ({allowed_prompt_length}) before generation.")


        try:
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            gen_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

            gen_output = gen_pipe(
                prompt,
                max_new_tokens=max_new_tokens_gen,
                num_return_sequences=1,
                do_sample=do_sample,
                temperature=0.7 if do_sample else 1.0,
                top_k=50 if do_sample else None,
                top_p=0.95 if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )

            if gen_output and isinstance(gen_output, list) and len(gen_output) > 0 and 'generated_text' in gen_output[0]:
                 generated_text = gen_output[0]['generated_text'].strip()
            else:
                 generated_text = ""
                 logger.warning(f"Pipeline returned unexpected output for prompt starting '{prompt[:50]}...'. Output: {gen_output}")

            if torch.cuda.is_available():
                 torch.cuda.empty_cache()

            final_answer = generated_text

            logger.info(f"Generated answer: {final_answer}")
            return final_answer

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return "Error: Could not generate answer."
