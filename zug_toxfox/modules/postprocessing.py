# type: ignore
import os
import re
import warnings

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from zug_toxfox import default_config, getLogger, pipeline_config
from zug_toxfox.utils import load_json, process_pollutants, remove_duplicates

log = getLogger(__name__)
warnings.filterwarnings("ignore")

config = pipeline_config.postprocessing


class FAISSIndexer:
    def __init__(self):
        self.model_name = config.FAISSIndexer_model_name
        self.model = SentenceTransformer(self.model_name)
        self.indices = {}
        self.key_tokens = {}
        self.use_gpu = faiss.get_num_gpus() > 0
        self.res = faiss.StandardGpuResources() if self.use_gpu else None

    def build_index(self, tokens: list[str], index_path: str) -> faiss.Index:
        if os.path.exists(index_path):
            log.info("Loading existing index...")
            index = faiss.read_index(index_path)
            if self.use_gpu:
                index = faiss.index_cpu_to_gpu(self.res, 0, index)
            return index

        log.info("Building new index...")
        embeddings = self.model.encode(tokens, convert_to_numpy=True, show_progress_bar=True)

        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])

        if self.use_gpu:
            index = faiss.index_cpu_to_gpu(self.res, 0, index)

        index.add(embeddings)

        log.info("Saving index to %s...", index_path)
        faiss.write_index(faiss.index_gpu_to_cpu(index) if self.use_gpu else index, index_path)
        return index

    def add_index(self, name: str, tokens: list[str], index_path: str) -> None:
        self.indices[name] = self.build_index(tokens, index_path)
        self.key_tokens[name] = tokens

    def search(self, name: str, queries: list[str], threshold: float) -> tuple[list[str], np.ndarray, np.ndarray]:
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)
        index = self.indices[name]
        distances, indices = index.search(query_embeddings, 1)
        result_tokens = np.array(self.key_tokens[name])[indices.flatten()]
        mask = distances.flatten() >= threshold
        return result_tokens, mask, distances


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False


class Trie:
    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()
        self.indexer = FAISSIndexer

    def insert(self, word: str) -> None:
        """Inserts a word into the Trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def non_matching(
        self, results: list[str], mask: list[bool], word: str, is_delimiter: str, i: int
    ) -> tuple[int, bool]:
        """Handles cases where no match in found in the Trie. Non-matching tokens are appended as a single string up to
        the next matching ingredient."""
        if ";" in word:
            cleaned_word = re.sub(r"[;]", "", word)
            if is_delimiter or not results:
                results.append(cleaned_word)
                mask.append(False)
            else:
                results[-1] += cleaned_word
            i += len(word) + 1
            is_delimiter = True
        elif is_delimiter:
            results.append(word)
            mask.append(False)
            i += len(word)
            is_delimiter = False
        elif not results or mask[-1]:
            word_split = word.split()
            if word_split:
                results.append(word_split[0])
                mask.append(False)
                i += len(word_split[0])
            else:
                i += 1
        else:
            results[-1] += word
            i += len(word)
        return i, is_delimiter

    def search(self, text: str) -> tuple[list[str], list[bool]]:
        """Searches for matching ingredients in the Trie."""
        results: list[str] = []
        mask: list[bool] = []
        i: int = 0
        is_delimiter = False
        while i < len(text):
            node: TrieNode = self.root
            word: str = ""
            longest_match: str = ""
            for n, char in enumerate(text[i:]):
                word += char
                if char in node.children:
                    node = node.children[char]
                    if node.is_end_of_word:
                        longest_match = word
                        if n == len(text[i:]) - 1:
                            results.append(longest_match.strip())
                            mask.append(True)
                            i += len(longest_match)
                            break
                else:
                    if longest_match:
                        results.append(longest_match.strip())
                        mask.append(True)
                        i += len(longest_match.strip()) + 1
                        if char == ";":
                            is_delimiter = False
                            i += 1
                    else:
                        i, is_delimiter = self.non_matching(results, mask, word, is_delimiter, i)
                    break

                if n == len(text[i:]) - 1:
                    i += 1

        return np.array([r.strip() for r in results]), np.array(mask)


class TokenCleaner:
    def __init__(self, indexer):
        self.indexer = indexer
        self.typo_threshold = config.typo_threshold
        self.misspelling_set = config.misspelling_set

    def clean_token(self, tokens: list[str]) -> list[str]:
        tokens = self.split_colon(tokens)
        cleaned_tokens = self.hyphen_and_parentheses(tokens)
        cleaned_tokens = [self.clean_word(word) for word in cleaned_tokens]
        return " ".join([token for token in cleaned_tokens for token in token.split() if len(token) > 1])

    def split_colon(self, tokens: list[str]) -> list[str]:
        words = []
        for token in tokens:
            parts = token.split(":")
            words.extend([part.strip() + (":" if i < len(parts) - 1 else "") for i, part in enumerate(parts)])
        return words

    def clean_word(self, word: str) -> str:
        cleaned_word = re.sub(r"[^\w\s():\\/]", "", word).strip()
        cleaned_word = re.sub(r"[^\w\s():\\/\-,;.]", "", word).strip()
        return self.correct_typos(cleaned_word.lower()) if cleaned_word else ""

    def hyphen_and_parentheses(self, split_token: list[str]) -> list[str]:
        length = len(split_token)

        cleaned_split_token = []
        skip_next = False

        for n in range(length):
            if skip_next:
                skip_next = False
                continue
            if split_token[n].endswith("-") and n + 1 < length:
                cleaned_split_token.append(split_token[n][:-1] + split_token[n + 1])
                skip_next = True
            elif (
                "(" in split_token[n]
                and ")" not in split_token[n]
                and n + 1 < length
                and ")" in split_token[n + 1]
                and "(" not in split_token[n + 1]
            ):
                cleaned_split_token.append(split_token[n] + " " + split_token[n + 1])
                skip_next = True
            else:
                cleaned_split_token.append(split_token[n])
                skip_next = False
        return cleaned_split_token

    def correct_typos(self, token: str) -> str:
        """Correct typos and misspelling of the OCR output using embedding search."""
        token = re.sub(r"\s+([,;.])", r"\1", token)

        queries = token.split()
        delimiter_mask = []
        delimiter_queries = []

        for q in queries:
            if any(delim in q for delim in [",", ";", "."]):
                delimiter_mask.append(True)
                q = re.sub(r"[,;.]", "", q)
            else:
                delimiter_mask.append(False)
            delimiter_queries.append(q)

        corrected_tokens, mask, _ = self.indexer.search("words", delimiter_queries, self.typo_threshold)
        delimiter_queries = np.array(delimiter_queries)
        delimiter_queries[mask] = corrected_tokens[mask].tolist()

        corrected_queries = [
            query if query.lower() not in self.misspelling_set else "oil" for query in delimiter_queries
        ]  # Correct typically misspelled word 'Oil'

        corrected_queries_with_delimiters = []
        for query, has_delimiter in zip(corrected_queries, delimiter_mask):  # noqa: B905
            if has_delimiter:
                query += ";"
            corrected_queries_with_delimiters.append(query)

        return " ".join(corrected_queries_with_delimiters)


class PostProcessor:
    def __init__(
        # TODO: Add type hints
        self,
        indexer: FAISSIndexer,
    ):
        self.ingredient_threshold = config.ingredient_threshold
        self.typo_threshold = config.typo_threshold
        self.misspelling_set = set(config.misspelling_set)
        self.detection_type = config.detection_type

        self.faiss_path = default_config.faiss_path
        self.pollutants_path = default_config.pollutants_path_simple
        self.inci_path = default_config.inci_path_simple
        self.synonym_path = default_config.synonym_path

        process_pollutants()

        with open(self.synonym_path) as file:
            self.synonym_to_ingredient = yaml.safe_load(file)

        self.indexer = indexer
        self.trie = Trie()

        self.known_ingredients = load_json(self.inci_path)

        if self.detection_type in ["pollutants", "both"]:
            self.pollutants = load_json(self.pollutants_path)

        self.combined_tokens = [ing.lower() for ing in self.known_ingredients]
        if self.detection_type in ["pollutants", "both"]:
            self.combined_tokens += remove_duplicates([pol.lower() for pol in self.pollutants])
        self.known_words = remove_duplicates([t.lower() for token in self.combined_tokens for t in token.split()])

        os.makedirs(self.faiss_path, exist_ok=True)
        self.indexer.add_index("tokens", self.combined_tokens, os.path.join(self.faiss_path, "faiss_tokens"))
        self.indexer.add_index("words", self.known_words, os.path.join(self.faiss_path, "faiss_words"))

        for ingredient in self.combined_tokens:
            self.trie.insert(ingredient)

        self.token_cleaner = TokenCleaner(indexer=self.indexer)

    def remove_redundancy(self, ingredients: list[str]) -> list[str]:
        corrected_ingredients = [ingredients[0]]
        for i in range(1, len(ingredients)):
            if not (
                i + 1 < len(ingredients)
                and ingredients[i] in ingredients[i - 1].split()
                and len(ingredients[i - 1].split()) > 1
            ):
                corrected_ingredients.append(ingredients[i])
        return corrected_ingredients

    def find_pollutants(self, synonym):
        return self.synonym_to_ingredient.get(synonym.strip(), None)

    def get_ingredients(self, tokens: list[str]) -> list[str]:
        cleaned_tokens = self.token_cleaner.clean_token(tokens)

        results = {"ingredients": [], "pollutants": []}
        trie_ingredients, mask_trie = self.trie.search(cleaned_tokens)
        remaining_tokens = trie_ingredients[mask_trie]

        if len(remaining_tokens) != 0:
            matches, mask, _ = self.indexer.search("tokens", remaining_tokens, self.ingredient_threshold)
            mask_corrected = np.bitwise_and(mask_trie[mask_trie], mask)

            idx_mask_trie = np.where(mask_trie)[0]
            trie_ingredients[idx_mask_trie[mask]] = matches[mask]

            mask_corrected_full = np.zeros_like(mask_trie, dtype=bool)
            mask_corrected_full[idx_mask_trie] = mask_corrected
            mask_trie[:] = mask_corrected_full

        ingredients = trie_ingredients[mask_trie]

        if len(ingredients) != 0:
            ingredients = self.remove_redundancy(remove_duplicates(ingredients))
            results["ingredients"] = ingredients

        if self.detection_type in ["pollutants", "both"] and ingredients:
            results["pollutants"] = remove_duplicates(
                [self.find_pollutants(ing) for ing in ingredients if self.find_pollutants(ing) is not None]
            )
        return results
