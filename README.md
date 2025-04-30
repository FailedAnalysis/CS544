# CS544
RAG QnA Chat-box: Smart Q&A using Retrieval and Generation
This project builds a command-line tool that answers questions by combining finding information and creating text. It uses outside knowledge to give better answers and avoid making things up.

What We Did
We built a full RAG (Retrieval Augmented Generation) system. Here are the main steps:

Built a Knowledge Base: We used all the text passages from the SQuAD 2.0 dataset to create a library of documents the system can search.

Trained a Strong Retriever:

We trained a model based on BERT to understand user questions.

This retriever quickly finds the most relevant text parts from our knowledge base for any question.

We added a step to re-sort the top results to make sure the best ones are at the top.

Fine-tuned a Text Generator:

We used the GPT-2 language model.

We fine-tuned GPT-2 using the SQuAD 2.0 dataset. This taught it how to create accurate answers using the question and the text found by the retriever.

Because we used SQuAD 2.0, our generator also learned how to handle questions that don't have an answer in the text.

Put the Retriever and Generator Together: We combined the trained retriever and generator. Now, when you ask a question, the system first finds relevant information, then uses that information to create the final answer.

Made a Command-Line Tool: We created a simple way to use the system by typing questions in a command window.

Project Goals and Results
Our goals for this project were to:

Improve Answer Quality: Use outside knowledge to give more accurate answers.

Handle New Information: Be able to use new knowledge by just updating the document library.

Handle Unanswerable Questions: Make the system recognize questions it can't answer from the knowledge base.

We successfully built a working command-line Q&A tool that takes your question, finds relevant info, and gives you an answer based on that info.

Training Details
Retriever Training: Trained a model based on SentenceTransformer to learn how questions and passages are related.

Generator Training: Standard fine-tuning of the GPT-2 model to generate answers given context.
