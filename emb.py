import os
import json
import torch
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
        base_url = os.getenv('API_EMBEDDER_BASE_URL')
        api_key = os.getenv('API_EMBEDDER_API_KEY')

        if not self.model:
            raise ValueError("API_EMBEDDER_MODEL_NAME not set in .env file")
        if not base_url:
            raise ValueError("API_EMBEDDER_BASE_URL not set in .env file")
        if not api_key:
            raise ValueError("API_EMBEDDER_API_KEY not set in .env file")

        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def get_emb_contriever(self, expansion_ids, expansion):
        all_docs_vectors = []

        # Process in batches to handle API rate limits
        batch_size = 64
        for i in tqdm(range(0, len(expansion), batch_size)):
            batch = expansion[i:i + batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )

                # Extract embeddings from response
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


def emb_rawdata(dataset, retriever):
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

    all_emb = []
    for conversation in tqdm(in_data):
        questions = [qa_item["question"] for qa_item in conversation["qa"]]
        sessions = ['\n'.join(session) for session in conversation["sessions"]]
        turns = [turn for session in conversation["sessions"] for turn in session]
        for sessid in conversation["sessions_ids"]:
            id = sessid if 'longmemeval' in dataset else f"convid-{str(conversation['conversation_id'])}-sessid-{sessid}"

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


if __name__ == '__main__':
    emb_rawdata('locomo10', 'contriever')
    # emb_rawdata('longmemeval_m', 'contriever')
    # locomo10, longmemeval_s, LongMTBench+
