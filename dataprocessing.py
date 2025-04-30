import json
import pandas as pd
import numpy as np
import torch
import re
import logging
import sys
import os
from tqdm.auto import tqdm

from torch.utils.data import DataLoader

from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sentence_transformers import SentenceTransformer, losses, models, util, InputExample, CrossEncoder
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import evaluate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


def load_json_data_squad(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'data' not in data:
             logger.error(f"JSON file {file_path} does not contain the expected 'data' key.")
             return None
        return data['data']
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {file_path}: {e}")
        return None

def process_squad_json(train_json_path, dev_json_path):
    logger.info(f"Loading and processing SQuAD 2.0 data from local files: {train_json_path}, {dev_json_path}")

    train_data = load_json_data_squad(train_json_path)
    dev_data = load_json_data_squad(dev_json_path)

    if train_data is None and dev_data is None:
         logger.error("Failed to load both train and dev JSON files.")
         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if train_data is None: train_data = []
    if dev_data is None: dev_data = []


    logger.info("Building corpus and context map...")
    context_to_id = {}
    documents_list = []
    doc_id_counter = 0

    def process_paragraphs_for_corpus(data_split, desc):
        nonlocal doc_id_counter
        if not isinstance(data_split, list): return
        for article in tqdm(data_split, desc=f"Processing articles ({desc} corpus)"):
            if not isinstance(article, dict) or 'paragraphs' not in article or not isinstance(article['paragraphs'], list): continue
            for paragraph in article['paragraphs']:
                if not isinstance(paragraph, dict) or 'context' not in paragraph: continue
                context = paragraph['context']
                if not isinstance(context, str):
                     logger.warning(f"Context is not a string in {desc}, type: {type(context)}. Skipping.")
                     continue
                stripped_context = context.strip()
                if stripped_context not in context_to_id:
                    doc_id = f"doc_{doc_id_counter}"
                    context_to_id[stripped_context] = doc_id
                    documents_list.append({'id': doc_id, 'passage': context})
                    doc_id_counter += 1

    process_paragraphs_for_corpus(train_data, "train")
    process_paragraphs_for_corpus(dev_data, "dev")

    documents_df = pd.DataFrame(documents_list)
    logger.info(f"Created corpus with {len(documents_df)} unique passages.")

    if documents_df.empty:
         logger.error("Corpus is empty after processing.")
         return documents_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    logger.info("Processing questions for Retriever and Generator DataFrames...")

    def process_qas_for_dataframes(data_split, desc):
        retriever_list = []
        generator_list = []

        if not isinstance(data_split, list): return ([], [])

        for article in tqdm(data_split, desc=f"Processing questions ({desc})"):
            if not isinstance(article, dict) or 'paragraphs' not in article or not isinstance(article['paragraphs'], list): continue
            for paragraph in article['paragraphs']:
                if not isinstance(paragraph, dict) or 'context' not in paragraph or 'qas' not in paragraph or not isinstance(paragraph['qas'], list): continue

                context = paragraph['context']
                context_str = str(context) if not isinstance(context, str) else context
                stripped_context = context_str.strip()
                passage_id = context_to_id.get(stripped_context)

                if passage_id is None:
                     logger.warning(f"Context '{stripped_context[:50]}...' not found in corpus map for a question in {desc}. Skipping questions in this paragraph.")
                     continue

                for qa in paragraph['qas']:
                    if not isinstance(qa, dict) or 'id' not in qa or 'question' not in qa or 'is_impossible' not in qa or 'answers' not in qa: continue

                    qid = str(qa['id'])
                    question_text = str(qa['question'])
                    is_impossible = qa['is_impossible']
                    answers = qa['answers']

                    retriever_list.append({
                        'id': qid,
                        'question': question_text,
                        'relevant_passage_ids': [passage_id]
                    })

                    answer_text = ""
                    if not is_impossible and answers and 'text' in answers and isinstance(answers['text'], list) and len(answers['text']) > 0:
                        answer_text = str(answers['text'][0])

                    generator_list.append({
                        'id': qid,
                        'question': question_text,
                        'answer': answer_text,
                        'relevant_passage_ids': [passage_id]
                    })
        return retriever_list, generator_list

    retriever_train_list, generator_train_list = process_qas_for_dataframes(train_data, "train")
    retriever_test_list, generator_dev_list = process_qas_for_dataframes(dev_data, "dev")

    retriever_train_df = pd.DataFrame(retriever_train_list)
    retriever_test_df = pd.DataFrame(retriever_test_list)
    generator_train_df = pd.DataFrame(generator_train_list)
    # Corrected variable name from generator_list to generator_dev_list
    generator_dev_df = pd.DataFrame(generator_dev_list)

    logger.info(f"Created retriever_train_df with {len(retriever_train_df)} questions.")
    logger.info(f"Created retriever_test_df with {len(retriever_test_df)} questions.")
    logger.info(f"Created generator_train_df with {len(generator_train_df)} questions.")
    logger.info(f"Created generator_dev_df with {len(generator_dev_df)} questions.")


    return documents_df, retriever_train_df, retriever_test_df, generator_train_df, generator_dev_df