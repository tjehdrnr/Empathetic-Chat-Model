import logging
from typing import List, Union, Dict

import faiss
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaissRetriever:

    def __init__(
        self,
        embedding_model,
        index,
        max_length=512, # max token length, default: 8192(BGE-M3)
        normalize_L2=True,
    ):
        self.embedding_model = embedding_model
        self.index = index
        self.max_length = max_length
        self.normalize_L2 = normalize_L2

        # Store each history's relative timestamp 
        self.relative_timestamps = np.array([], dtype=np.float64)

        self.current_history_id = 0


    def _query_to_vector(self, query: Union[List[str], str]) -> np.ndarray:
        """
        Embed queries into vectors.
        Queries are divided into two types: user input or history.
        """
        if isinstance(query, str):
            query = [query]

        # Embed user input or latest history
        vectors = self.embedding_model.encode(
            query, max_length=self.max_length, return_dense=True,
            )['dense_vecs']
        
        vectors = np.array(vectors, dtype=np.float32)
        if self.normalize_L2:
            faiss.normalize_L2(vectors)

        return vectors
    

    def add_to_index(self, latest_history: Dict) -> None:
        """
        Add history vectors to the Faiss index.
        """
        vectors = self._query_to_vector(latest_history['text'])
        self.index.add_with_ids(vectors, np.array([self.current_history_id]))

        self.relative_timestamps = np.append(
            self.relative_timestamps, latest_history['relative_time'],
        )

        self.current_history_id += 1

        logger.info(f"Added {vectors.shape[0]} history vector to the index")
        logger.info(f"Added {latest_history['relative_time']}/sec to relative timestamps")
    

    def remove_id(self, target_history_index: int) -> None:
        """
        Remove stored vector in Faiss index.
        """
        self.index.remove_ids(np.array([target_history_index]))     


    def search_similar(self, query: Union[List[str], str], **search_kwargs) -> tuple:
        """
        Searches for vectors similar to user inputs and returns the search results
        In the process of finding similar vectors, I used a weighted sum of time weight in the process.
        This prevents retrieving history that goes too far in the past.
        """
        if isinstance(query, str):
            query = [query]
        
        k = search_kwargs.pop('k', 2)
        time_weight = search_kwargs.pop('time_weight', 0.2)
        
        query_vector = self._query_to_vector(query)
        distances, indices = self.index.search(query_vector, k)

        if len(self.relative_timestamps) < 2:
            return distances[0], indices[0]
        
        # Apply time weight to similarity calculation
        current_time = np.max(self.relative_timestamps)
        time_diffs = current_time - self.relative_timestamps[indices[0]]
        max_time_diff = current_time - np.min(self.relative_timestamps)

        if max_time_diff == 0:
            return distances[0], indices[0]

        # Normalize time weights. The closer to 0, the more recent history
        normd_time_weights = time_diffs / max_time_diff

        # Calculate time weighted distances (Combination of vector weights and time weights)
        time_weighted_distances = (1 - time_weight) * distances[0] + time_weight * normd_time_weights

        # Reorder results
        sorted_indices = np.argsort(time_weighted_distances)
        distances = time_weighted_distances[sorted_indices]
        indices = indices[0][sorted_indices]

        return distances, indices
    
    
    def search_similar_without_time(self, query: Union[List[str], str], **search_kwargs) -> tuple:
        """
        When DPO mode is activated, use this method instead of the 'search_similar' method.
        """
        if isinstance(query, str):
            query = [query]
        
        k = search_kwargs.pop('k', 2)

        query_vector = self._query_to_vector(query)
        distances, indices = self.index.search(query_vector, k)

        return distances, indices[0]


    @staticmethod
    def print_results(query, distances, indices, docstore) -> None:
        """
        Validation for time weighted search.
        """
        print(f"\nUser query: {query}")
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            print(f"{i+1}. Distance: {dist}, Index: {idx}")
            print(f"Content: {docstore.history[idx]['text']}")
