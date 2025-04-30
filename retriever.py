class Retriever:

    def __init__(self, retriever_model_name="all-mpnet-base-v2", reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Initializing retriever model: {retriever_model_name}")
        self.retriever_model = SentenceTransformer(retriever_model_name)

        logger.info(f"Initializing reranker model: {reranker_model_name}")
        self.reranker_model = CrossEncoder(reranker_model_name)

        self.corpus = {}
        self.corpus_ids = []
        self.corpus_texts = []
        self.corpus_embeddings = None
        self.corpus_embeddings_device = None

        self.train_examples = []
        self.queries = {}
        self.relevant_docs = {}

    def load_corpus(self, documents_df):
        logger.info("Loading corpus documents...")
        documents_df['id'] = documents_df['id'].astype(str)
        documents_df['passage'] = documents_df['passage'].fillna("").astype(str)

        self.corpus = dict(zip(documents_df['id'], documents_df['passage']))
        self.corpus_ids = list(self.corpus.keys())
        self.corpus_texts = list(self.corpus.values())

        logger.info(f"Corpus size: {len(self.corpus)} documents.")

    def prepare_data(self, train_df, negative_samples=3, eval_ratio=0.2):
        if not self.corpus:
            raise ValueError("No corpus loaded. Call load_corpus(documents_df) first.")

        logger.info(f"Preparing training data from DataFrame ({len(train_df)} rows)...")

        train_df['id'] = train_df['id'].astype(str)
        train_df['question'] = train_df['question'].fillna("").astype(str)
        train_df['relevant_passage_ids'] = train_df['relevant_passage_ids'].apply(
             lambda x: [str(i).strip() for i in (eval(str(x)) if isinstance(x, str) else x) if str(i).strip()] if pd.notna(x) else []
        )


        total_len = len(train_df)
        eval_size = int(eval_ratio * total_len)
        eval_size = min(eval_size, total_len)
        if total_len > 0 and eval_ratio > 0 and eval_size == 0: eval_size = 1
        eval_indices = set(np.random.choice(range(total_len), size=eval_size, replace=False)) if eval_size > 0 else set()

        train_examples = []
        queries = {}
        relevant_docs = {}

        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Preparing data"):
            qid = str(row['id'])
            question_text = row['question']
            rel_ids = row['relevant_passage_ids']

            valid_rel_ids = [rid for rid in rel_ids if rid in self.corpus]
            for rid in valid_rel_ids:
                 train_examples.append(InputExample(
                     texts=[question_text, self.corpus[rid]],
                     label=1.0
                 ))
            if not valid_rel_ids:
                 logger.warning(f"Query {qid} has no relevant documents found in corpus. Cannot create positive examples for training.")


            if valid_rel_ids:
                all_irrelevant_ids = [pid for pid in self.corpus.keys() if pid not in rel_ids]
                if len(all_irrelevant_ids) > 0 and negative_samples > 0:
                    neg_sample_ids = np.random.choice(
                        all_irrelevant_ids,
                        min(negative_samples, len(all_irrelevant_ids)),
                        replace=False
                    )
                    for nid in neg_sample_ids:
                        train_examples.append(InputExample(
                            texts=[question_text, self.corpus[nid]],
                            label=0.0
                        ))

            if idx in eval_indices:
                 queries[qid] = question_text
                 relevant_docs[qid] = {doc_id: 1 for doc_id in rel_ids if doc_id in self.corpus}
                 if not relevant_docs[qid]:
                      if qid in queries: del queries[qid]
                      if qid in relevant_docs: del relevant_docs[qid]


        self.train_examples = train_examples
        self.queries = queries
        self.relevant_docs = relevant_docs

        logger.info(f"Total training examples: {len(self.train_examples)}")
        logger.info(f"Eval queries: {len(self.queries)}; corpus size: {len(self.corpus)}")


    def train(self,
              epochs=1,
              evaluation_steps=250,
              warmup_steps=200,
              output_path="output/retriever-model"):
        if not self.train_examples:
            raise ValueError("No training examples found. Run prepare_data(...) first.")

        logger.info(f"Training retriever model on {len(self.train_examples)} examples...")
        train_batch_size = 16
        train_dataloader = DataLoader(self.train_examples, batch_size=train_batch_size, shuffle=True)
        train_loss = losses.MultipleNegativesRankingLoss(self.retriever_model)

        ir_evaluator = None
        if evaluation_steps > 0 and len(self.queries) > 0 and len(self.corpus) > 0 and len(self.relevant_docs) > 0:
            logger.info("Setting up Information Retrieval Evaluator...")
            ir_evaluator = InformationRetrievalEvaluator(
                queries={qid: str(q) for qid, q in self.queries.items()},
                corpus={doc_id: str(doc) for doc_id, doc in self.corpus.items()},
                relevant_docs=self.relevant_docs,
                show_progress_bar=True,
                corpus_chunk_size=100000
            )
        elif evaluation_steps > 0:
             logger.warning(f"Skipping evaluation during training (evaluation_steps > 0) as eval data is incomplete.")
             evaluation_steps = 0


        logger.info("Starting training...")
        self.retriever_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=ir_evaluator,
            epochs=epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True
        )
        logger.info("Retriever training complete!")


    def fit(self,
            train_df: pd.DataFrame,
            negative_samples=3,
            eval_ratio=0.2,
            epochs=1,
            evaluation_steps=250,
            warmup_steps=200,
            output_path="output/retriever-model"):
        self.prepare_data(train_df, negative_samples=negative_samples, eval_ratio=eval_ratio)
        self.train(epochs=epochs,
                   evaluation_steps=evaluation_steps,
                   warmup_steps=warmup_steps,
                   output_path=output_path)


    def precompute_corpus_embeddings(self, batch_size=1024):
        if not self.corpus_texts:
            raise ValueError("No corpus found. Did you run load_corpus(...) first.")

        logger.info(f"Computing embeddings for {len(self.corpus_texts)} passages...")
        all_embeddings = []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Encoding device: {device}")
        self.retriever_model.to(device)

        try:
            for start_idx in tqdm(range(0, len(self.corpus_texts), batch_size), desc="Encoding corpus"):
                batch = self.corpus_texts[start_idx:start_idx + batch_size]
                batch = [str(t) for t in batch]
                batch_embeddings = self.retriever_model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=device
                )
                all_embeddings.append(batch_embeddings)

            self.corpus_embeddings = torch.cat(all_embeddings, dim=0)
            self.corpus_embeddings_device = self.corpus_embeddings.device
            logger.info(f"Corpus embeddings shape: {self.corpus_embeddings.shape} on device: {self.corpus_embeddings_device}")

        except Exception as e:
            logger.error(f"Error during corpus encoding: {e}")
            self.corpus_embeddings = None
            self.corpus_embeddings_device = None
            raise


    def re_rank_passages(self, query, initial_results):
        if not initial_results: return []
        logger.info(f"Re-ranking {len(initial_results)} passages for query...")
        cross_encoder_input = [[str(query), str(res['passage'])] for res in initial_results]
        if not cross_encoder_input: return []

        try:
            rerank_scores = self.reranker_model.predict(cross_encoder_input)
            for i, res in enumerate(initial_results):
                res['rerank_score'] = float(rerank_scores[i])
            reranked_results = sorted(initial_results, key=lambda x: x.get('rerank_score', -float('inf')), reverse=True)
            logger.info("Re-ranking complete.")
            return reranked_results
        except Exception as e:
            logger.error(f"Error during re-ranking: {e}")
            logger.warning("Re-ranking failed. Returning initial results sorted by original score.")
            return sorted(initial_results, key=lambda x: x.get('score', -float('inf')), reverse=True)


    def retrieve_top_k(self, query, top_k=5, initial_retrieval_k=100, use_reranking=True):
        if self.corpus_embeddings is None or self.corpus_embeddings_device is None:
            logger.info("Corpus embeddings not precomputed or device unknown. Computing now.")
            self.precompute_corpus_embeddings()
            if self.corpus_embeddings is None or self.corpus_embeddings_device is None:
                 logger.error("Failed to compute corpus embeddings. Cannot perform retrieval.")
                 return []

        device = self.corpus_embeddings_device
        self.retriever_model.to(device)
        query_embedding = self.retriever_model.encode(str(query), convert_to_tensor=True, device=device)

        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        initial_k = initial_retrieval_k if use_reranking else top_k
        initial_k = min(initial_k, len(self.corpus_texts))
        initial_k = max(1, initial_k) if len(self.corpus_texts) > 0 else 0

        if initial_k == 0:
             logger.warning("Corpus is empty or initial_k is 0. Cannot retrieve.")
             return []

        top_values, top_indices = torch.topk(cos_scores, k=initial_k)
        top_indices_list = top_indices.cpu().numpy()

        initial_retrieved_passages = []
        corpus_ids_str = [str(cid) for cid in self.corpus_ids]

        for idx in top_indices_list:
            cid = corpus_ids_str[idx]
            passage = self.corpus.get(cid, "")
            initial_retrieved_passages.append({
                'corpus_id': cid,
                'passage': passage,
                'score': cos_scores[idx].item()
            })

        if use_reranking:
            reranked_results = self.re_rank_passages(query, initial_retrieved_passages)
            return reranked_results[:top_k]
        else:
            return sorted(initial_retrieved_passages, key=lambda x: x.get('score', -float('inf')), reverse=True)[:top_k]


    def retrieve_for_test(self, test_df, top_k=5, initial_retrieval_k=100, use_reranking=True):
        if self.corpus_embeddings is None or self.corpus_embeddings_device is None:
            logger.info("Corpus embeddings not precomputed or device unknown. Computing now.")
            self.precompute_corpus_embeddings()
            if self.corpus_embeddings is None or self.corpus_embeddings_device is None:
                 logger.error("Failed to compute corpus embeddings. Cannot perform retrieval for test set.")
                 result_df = test_df.copy()
                 result_df['relevant_passage_ids'] = [[]] * len(test_df)
                 return result_df

        if test_df.empty:
             logger.warning("Test DataFrame is empty. Skipping retrieval for test set.")
             return test_df.copy().assign(relevant_passage_ids=[[] for _ in range(len(test_df))])


        queries = test_df['question'].astype(str).tolist()

        device = self.corpus_embeddings_device
        self.retriever_model.to(device)
        query_embeddings = self.retriever_model.encode(queries, convert_to_tensor=True, device=device, show_progress_bar=True)

        cos_scores = util.cos_sim(query_embeddings, self.corpus_embeddings)

        relevant_passage_ids = []

        initial_k = initial_retrieval_k if use_reranking else top_k
        initial_k = min(initial_k, len(self.corpus_texts))
        initial_k = max(1, initial_k) if len(self.corpus_texts) > 0 else 0

        if initial_k == 0:
             logger.warning("Corpus is empty or initial_k is 0. Cannot retrieve for test set.")
             relevant_passage_ids = [[]] * len(test_df)
        else:
            corpus_ids_str = [str(cid) for cid in self.corpus_ids]

            for i in tqdm(range(len(test_df)), desc="Retrieving for test set"):
                row_scores = cos_scores[i]
                top_values, top_indices = torch.topk(row_scores, k=initial_k)
                top_indices_list = top_indices.cpu().numpy()

                initial_retrieved_passages = []
                for idx in top_indices_list:
                     cid = corpus_ids_str[idx]
                     passage = self.corpus.get(cid, "")
                     initial_retrieved_passages.append({
                         'corpus_id': cid,
                         'passage': passage,
                         'score': row_scores[idx].item()
                     })

                if use_reranking:
                    query_text = queries[i]
                    reranked_results = self.re_rank_passages(query_text, initial_retrieved_passages)
                    top_k_ids = [res['corpus_id'] for res in reranked_results[:top_k]]
                else:
                    top_k_ids = [res['corpus_id'] for res in sorted(initial_retrieved_passages, key=lambda x: x.get('score', -float('inf')), reverse=True)[:top_k]]

                relevant_passage_ids.append(top_k_ids)

        result_df = test_df.copy()
        result_df['relevant_passage_ids'] = relevant_passage_ids
        return result_df


    def evaluate(self, test_df, top_k=5, metrics_k=10, initial_retrieval_k=100, use_reranking=True):
        if metrics_k > top_k:
             logger.warning(f"metrics_k ({metrics_k}) is greater than top_k ({top_k}). Metrics will be calculated based on top_{top_k} results.")
             metrics_k = top_k
        if metrics_k <= 0:
             logger.warning("metrics_k is <= 0. Skipping evaluation.")
             return {}
        if test_df.empty or 'relevant_passage_ids' not in test_df.columns:
             logger.warning("Test DataFrame is empty or missing 'relevant_passage_ids' for evaluation. Skipping evaluation.")
             return { f"recall@{metrics_k}": 0.0, f"precision@{metrics_k}": 0.0, "mrr": 0.0 }


        logger.info(f"Starting evaluation with metrics_k={metrics_k} (top_k={top_k}, use_reranking={use_reranking})")

        test_df['relevant_passage_ids'] = test_df['relevant_passage_ids'].apply(lambda x: [str(i).strip() for i in (eval(str(x)) if isinstance(x, str) else x) if str(i).strip()] if pd.notna(x) else []
        )

        retrieved_df = self.retrieve_for_test(test_df[['id', 'question']].copy(),
                                              top_k=top_k,
                                              initial_retrieval_k=initial_retrieval_k,
                                              use_reranking=use_reranking)

        results_dict = {}
        relevant_dict = {}

        test_df['id'] = test_df['id'].astype(str)
        retrieved_df['id'] = retrieved_df['id'].astype(str)

        predicted_ids_map = dict(zip(retrieved_df['id'], retrieved_df['relevant_passage_ids']))

        common_query_ids = set(test_df['id']).intersection(set(retrieved_df['id']))
        if not common_query_ids:
             logger.warning("No common query IDs between original test_df and retrieved_df. Cannot compute metrics.")
             return { f"recall@{metrics_k}": 0.0, f"precision@{metrics_k}": 0.0, "mrr": 0.0 }

        filtered_test_df = test_df[test_df['id'].isin(common_query_ids)].set_index('id')
        filtered_retrieved_df = retrieved_df[retrieved_df['id'].isin(common_query_ids)].set_index('id')


        for qid in tqdm(common_query_ids, desc="Processing queries for metrics"):
            predicted_list = filtered_retrieved_df.loc[qid, 'relevant_passage_ids']
            results_dict[qid] = predicted_list[:metrics_k]

            true_ids = filtered_test_df.loc[qid, 'relevant_passage_ids']
            relevant_dict[qid] = {doc_id: 1 for doc_id in true_ids if doc_id in self.corpus}

            if not relevant_dict[qid] and qid in results_dict:
                 del results_dict[qid]

        return self.evaluate_ir_metrics(results_dict, relevant_dict, k=metrics_k)


    def evaluate_ir_metrics(self, results, relevant_docs, k=10):
        common_qids = set(results.keys()).intersection(set(relevant_docs.keys()))

        if not common_qids:
             logger.warning("Cannot compute IR metrics: No common queries with both predictions and ground truth relevant docs in corpus.")
             return { f"recall@{k}": 0.0, f"precision@{k}": 0.0, "mrr": 0.0 }

        filtered_results = {qid: results[qid] for qid in common_qids}
        filtered_relevant_docs = {qid: relevant_docs[qid] for qid in common_qids}

        recall = self._calculate_recall_at_k(filtered_results, filtered_relevant_docs, k)
        precision = self._calculate_precision_at_k(filtered_results, filtered_relevant_docs, k)
        mrr = self._calculate_mrr(filtered_results, filtered_relevant_docs)

        return {
            f"recall@{k}": recall,
            f"precision@{k}": precision,
            "mrr": mrr
        }

    def _calculate_recall_at_k(self, results, relevant_docs, k):
        recalls = []
        for query_id, retrieved_docs in results.items():
            if query_id in relevant_docs:
                relevant = set(relevant_docs[query_id].keys())
                retrieved = set(retrieved_docs if isinstance(retrieved_docs, list) else [])
                if len(relevant) > 0:
                    recall = len(relevant.intersection(retrieved)) / len(relevant)
                    recalls.append(recall)
        return sum(recalls) / len(recalls) if recalls else 0.0

    def _calculate_precision_at_k(self, results, relevant_docs, k):
        precisions = []
        for query_id, retrieved_docs in results.items():
            if query_id in relevant_docs:
                relevant = set(relevant_docs[query_id].keys())
                retrieved = retrieved_docs if isinstance(retrieved_docs, list) else []
                if len(retrieved) > 0:
                    precision = len(relevant.intersection(set(retrieved))) / len(retrieved)
                    precisions.append(precision)
                else:
                    precisions.append(0.0)
        return sum(precisions) / len(precisions) if precisions else 0.0

    def _calculate_mrr(self, results, relevant_docs):
        mrr_scores = []
        for query_id, retrieved_docs in results.items():
            if query_id in relevant_docs:
                 relevant = set(relevant_docs[query_id].keys())
                 if isinstance(retrieved_docs, list):
                     found_relevant = False
                     for i, doc_id in enumerate(retrieved_docs):
                         if doc_id in relevant:
                             mrr_scores.append(1.0 / (i + 1))
                             found_relevant = True
                             break
                     if not found_relevant:
                         mrr_scores.append(0.0)
                 else:
                      mrr_scores.append(0.0)
        return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0