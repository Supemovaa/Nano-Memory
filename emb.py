import os
import json
import torch
import asyncio
import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def read_ids2granularity(path):
    ids2granularity = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())  
            ids2granularity.update(data)
    return ids2granularity
    

class EmbeddingModelContriever():

    def __init__(self):
        contriever_path = os.getenv('CONTRIEVER_PATH', "")
        self.model = AutoModel.from_pretrained(contriever_path).to(torch.device('cuda', 0))
        self.tokenizer = AutoTokenizer.from_pretrained(contriever_path)

    def get_emb_contriever(self, expansion_ids, expansion):
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        
        with torch.no_grad():
            all_docs_vectors = []
            dataloader = DataLoader(expansion, batch_size=64, shuffle=False)
            for batch in tqdm(dataloader):
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                cur_docs_vectors = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                all_docs_vectors.append(cur_docs_vectors)
            all_docs_vectors = torch.concat(all_docs_vectors, axis=0)
        if expansion_ids:
            ids2emb = {expansion_ids[i]: all_docs_vectors[i] for i in range(len(expansion_ids))}
            return ids2emb
        else:
            return all_docs_vectors


class EmbeddingModelSBERT():
    def __init__(self, retriever):
        if retriever == 'mpnet':
            self.model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
        elif retriever == 'minilm':
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def get_emb_contriever(self, expansion_ids, expansion):
        all_docs_vectors = self.model.encode(expansion)
        if expansion_ids:
            ids2emb = {expansion_ids[i]: all_docs_vectors[i] for i in range(len(expansion_ids))}
            return ids2emb
        else:
            return torch.tensor(all_docs_vectors)


class EmbeddingModelAPI():
    def __init__(self):
        self.model = os.getenv('API_EMBEDDER_MODEL_NAME')
        self.base_url = os.getenv('API_EMBEDDER_BASE_URL')
        self.api_key = os.getenv('API_EMBEDDER_API_KEY')

        if not self.model:
            raise ValueError("API_EMBEDDER_MODEL_NAME not set in .env file")
        if not self.base_url:
            raise ValueError("API_EMBEDDER_BASE_URL not set in .env file")
        if not self.api_key:
            raise ValueError("API_EMBEDDER_API_KEY not set in .env file")

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    async def get_embeddings_async(self, session, semaphore, batch):
        """Async function to get embeddings for a batch with concurrency control"""
        async with semaphore:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            retry = 0
            while retry < 10:
                try:
                    async with session.post(
                        url=f"{self.base_url}/embeddings",
                        json={
                            "model": self.model,
                            "input": batch
                        },
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            batch_embeddings = [np.array(item['embedding']) for item in result['data']]
                            return batch_embeddings
                        else:
                            error_text = await response.text()
                            print(f"API Error {response.status}: {error_text}")
                            retry += 1
                            if retry < 10:
                                await asyncio.sleep(1)
                except Exception as e:
                    retry += 1
                    if retry < 10:
                        await asyncio.sleep(1)
                        print(f"Error: {e}, retry {retry}/10")
                    else:
                        print(f"Failed after 10 retries: {e}")
                        raise

    def get_emb_contriever(self, expansion_ids, expansion):
        """Synchronous wrapper for backward compatibility"""
        all_docs_vectors = []

        batch_size = 64
        for i in tqdm(range(0, len(expansion), batch_size)):
            batch = expansion[i:i + batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_docs_vectors.extend(batch_embeddings)

            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                raise

        all_docs_vectors = torch.tensor(np.array(all_docs_vectors))

        if expansion_ids:
            ids2emb = {expansion_ids[i]: all_docs_vectors[i] for i in range(len(expansion_ids))}
            return ids2emb
        else:
            return all_docs_vectors


async def emb_rawdata_async(dataset, retriever, max_concurrent=32):
    """Async version with global 32-way parallelism for API calls"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    emb_dir = os.path.join(script_dir, 'logs/process_embs')
    os.makedirs(emb_dir, exist_ok=True)
    save_path = os.path.join(emb_dir, f'{dataset}-{retriever}-emb.pt')
    data_path = f'data/process_data/{dataset}.json'
    in_data = json.load(open(data_path))

    if retriever == 'contriever':
        emb_model = EmbeddingModelContriever()
    elif retriever in ['mpnet', 'minilm']:
        emb_model = EmbeddingModelSBERT(retriever)
    elif retriever == 'api':
        emb_model = EmbeddingModelAPI()
    else:
        raise ValueError(f"Unknown retriever: {retriever}")

    # For non-API models, use synchronous path
    if retriever != 'api':
        all_emb = []
        for conversation in tqdm(in_data):
            questions = [qa_item["question"] for qa_item in conversation["qa"]]
            sessions = ['\n'.join(session) for session in conversation["sessions"]]
            turns = [turn for session in conversation["sessions"] for turn in session]

            q_embs = emb_model.get_emb_contriever(None, questions)
            s_embs = emb_model.get_emb_contriever(None, sessions)
            t_embs = emb_model.get_emb_contriever(None, turns)

            embform = {
                "conversation_id": conversation['conversation_id'],
                "questions": q_embs,
                "sessions": s_embs,
                "turns": t_embs,
            }
            all_emb.append(embform)

        torch.save(all_emb, save_path)
        return all_emb

    # API model: use async with global concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    # Prepare all batches from all conversations
    batch_size = 64
    tasks = []
    batch_metadata = []  # Track which conversation and type each batch belongs to

    for conv_idx, conversation in enumerate(in_data):
        questions = [qa_item["question"] for qa_item in conversation["qa"]]
        sessions = ['\n'.join(session) for session in conversation["sessions"]]
        turns = [turn for session in conversation["sessions"] for turn in session]

        # Split each into batches and create tasks
        for text_list, text_type in [(questions, 'questions'), (sessions, 'sessions'), (turns, 'turns')]:
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i:i + batch_size]
                batch_metadata.append({
                    'conv_idx': conv_idx,
                    'type': text_type,
                    'start_idx': i,
                    'batch': batch
                })

    print(f"Total batches to process: {len(batch_metadata)}")

    # Execute all batches concurrently with global limit of 32
    async with aiohttp.ClientSession() as session:
        tasks = [
            emb_model.get_embeddings_async(session, semaphore, meta['batch'])
            for meta in batch_metadata
        ]

        # Show progress while gathering
        print("Processing batches with 32-way parallelism...")
        results = await asyncio.gather(*tasks)

    # Reconstruct embeddings per conversation
    all_emb = [None] * len(in_data)
    conv_embeddings = {i: {'questions': [], 'sessions': [], 'turns': []} for i in range(len(in_data))}

    for meta, result in zip(batch_metadata, results):
        conv_idx = meta['conv_idx']
        text_type = meta['type']
        conv_embeddings[conv_idx][text_type].extend(result)

    # Convert to tensors and create final structure
    for conv_idx, conversation in enumerate(in_data):
        q_embs = torch.tensor(np.array(conv_embeddings[conv_idx]['questions']))
        s_embs = torch.tensor(np.array(conv_embeddings[conv_idx]['sessions']))
        t_embs = torch.tensor(np.array(conv_embeddings[conv_idx]['turns']))

        embform = {
            "conversation_id": conversation['conversation_id'],
            "questions": q_embs,
            "sessions": s_embs,
            "turns": t_embs,
        }
        all_emb[conv_idx] = embform

    torch.save(all_emb, save_path)
    return all_emb


def emb_rawdata(dataset, retriever):
    """Synchronous wrapper that calls async version"""
    return asyncio.run(emb_rawdata_async(dataset, retriever, max_concurrent=32))


if __name__ == '__main__':
    emb_rawdata('locomo10', 'contriever')
    # emb_rawdata('longmemeval_m', 'contriever')
    # locomo10, longmemeval_s, LongMTBench+
