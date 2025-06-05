""" Utilities for calculating semantic similarity with LaBSE embeddings and filtering based on this score  """
from sentence_transformers import SentenceTransformer
from torch import vstack

class SimilarityScorer:
    def __init__(self, device: str = None, batch_size: int = 64):
        self.model = SentenceTransformer(
            "sentence-transformers/LaBSE",
            local_files_only = True,
            device = device
        )
        self.batch_size = batch_size
    
    def similarity(self, src: str, tgt: str | list[str]) -> "torch.Tensor":
        """Calulcate the similarity between a single source sentence
        and one ore more target sentences. 

        Arguments:
            src (str): A single source string
            tgt (str | list[str]): A single target string or a list of target strings
        
        Returns: 
            A (1, len(tgt)) tensor where r[0][j] is the cosine similarity between the src and tgt[j]
        
        """
        batch = [src] + tgt if isinstance(tgt, list) else [tgt]
        emb = self.model.encode(batch, batch_size = self.batch_size, show_progress_bar = False)
        
        sim = self.model.similarity(emb[0], emb[1:])

        return sim

    
    def similarity_batch(self, sources: list[str], targets: list[list[str]]) -> "torch.Tensor":
        """Calulcate the similarity between a a batch of single source sentences
        associated with one ore more target sentences. 

        Arguments:
            sources (list[str]): A list of N source sentences
            targets (list[list[str]]): A list of size NxM, where M is the number of targets per source
        
        Returns: 
            A (N, M) tensor r where r[i][j] is the cosine similarity between the sources[i] and targets[i][j]
        
        """
        
        N = len(sources)
        M = len(targets[0])

        
        embeddings = []
        merged_sentences = sources + [t for target in targets for t in target]
        emb = self.model.encode(merged_sentences, batch_size = self.batch_size, show_progress_bar = False) # -> dim(N+N*M, d_model)

        sims = [self.model.similarity(emb[i], emb[i*M+N : i*M+N+M]) for i in range(N)] # N * dim(1xM)
        sim = vstack(sims) # dim(N, M)

        return sim

    
    def filter(self, src: str, tgts: list[str], top_k: int = 2, threshold: float = 0.8) -> list[str]:
        """Filter hypotheses by computing cosine similarity between the source and 
        the hypotheses and returning the top k sentences provided they are above the threshold.
        
        Arguments:
            src (str): A single source string
            tgt (str | list[str]): A single target string or a list of target strings
            top_k (int): Maximum number of translations to return
            threshold (float): Only return translations with similarity score above this threshold

        Returns:
            A list containing between 0 and top_k hypotheses
        """
        sim = self.similarity(src, tgts) # (1, M)
        top = sim[0].topk(top_k) # topk (1, top_k)
        mask = top.values > threshold # (1, top_k)

        return [tgts[idx] for i, idx in enumerate(top.indices) if mask[i]]

    
    def filter_batch(self, sources: list[str], targets: list[list[str]], top_k: int = 2, threshold: float = 0.8) -> list[list[str]]:
        """Filter hypotheses in batches by computing cosine similarity between the source and 
        the hypotheses and returning the top k sentences provided they are above the threshold.
        
        Arguments:
            sources (list[str]): A list of N source sentences
            targets (list[list[str]]): A list of size (N, M), where M is the number of targets per source
            top_k (int): Maximum number of translations to return
            threshold (float): Only return translations with similarity score above this threshold

        Returns:
            A list of lists of size (N, 0->top_k) such that for each source sentence, the corresponding list
            contains between 0 and top_k hypotheses depending on the the threshold filtering
        """
        sim = self.similarity_batch(sources, targets) # (N, M)
        top = sim.topk(top_k, dim = 1) # topk (N, top_k)
        mask = top.values > threshold # (N, top_k)

        """filtered = []
        # Each row
        for i, idx in enumerate(top.indices):
            # Each column e.g. idx = Tensor([0,2])
            row = []
            for j, jdx in enumerate(idx):
                if mask[i][j]:
                    row.append(targets[i][jdx.item()])
            filtered.append(row)"""

        filtered = [
            [targets[i][jdx.item()] for j, jdx in enumerate(idx) if mask[i][j]]
            for i, idx in enumerate(top.indices)
        ]
        
        #[targets[i][jdx.item()] for i, idx in enumerate(top.indices) for j, jdx in enumerate(idx) if mask[i][j]]
        return filtered # dim (N, range(0, top_k))

if __name__ == "__main__":
    scorer = SimilarityScorer()
