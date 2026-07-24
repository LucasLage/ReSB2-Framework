from opensearchpy import OpenSearch, helpers
import pandas as pd
from tqdm import tqdm

class OpenSearchOperator:
    """
    This class encapsulates the initialization of an OpenSearch client, handling
    authentication and host configuration. It provides a centralized entry point
    for performing search, indexing, and administrative operations.

    Attributes
    ----------
    client : OpenSearch
        The initialized OpenSearch client instance, configured with host, port,
        credentials, and certificate settings.

    Parameters
    ----------
    host : str
        The hostname or IP address of the OpenSearch cluster.
    port : int
        The port number where OpenSearch is running.
    user : str
        Username for HTTP authentication.
    password : str
        Password for HTTP authentication.
    """

    def __init__(self, host, port, user, password):
        """
        Initializes an OpenSearch client with the given connection parameters.

        This method configures the client with host and port information,
        applies HTTP authentication, and disables certificate verification.

        Parameters
        ----------
        host : str
            The hostname or IP address of the OpenSearch instance.
        port : int
            The port on which the OpenSearch service is running.
        user : str
            The username used for HTTP authentication.
        password : str
            The password used for HTTP authentication.
        """
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(user, password),
            verify_certs=False
        )

    def create_index(self, index_name, embedding_dimension):
        """
        Creates a new OpenSearch index configured for vector search.

        If an index with the same name already exists, it is deleted before
        creation. The method defines index settings enabling k-NN search and
        configures mappings for metadata fields and an embedding vector field.

        Parameters
        ----------
        index_name : str
            Name of the index to be created.
        embedding_dimension : int
            Dimensionality of the embedding vectors stored in the 'embedding' field.
        """
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)

        index_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "texto": {"type": "text"},
                    "legislature": {"type": "integer"},
                    "legislative_session": {"type": "integer"},
                    "embedding": {"type": "knn_vector", "dimension": embedding_dimension}
                }
            }
        }

        self.client.indices.create(index=index_name, body=index_body)

    def index_data(self, index_name, df, model, embedding_by_index):
        """
        Indexes textual and vector data into an OpenSearch index.

        This method iterates through a pandas DataFrame, generating documents
        that include metadata, text, and embeddings. Embeddings can be obtained
        either from a precomputed lookup table (`embedding_by_index`) or
        generated on the fly using a sentence embedding model. Documents are
        sent to OpenSearch in bulk for efficient ingestion.

        Parameters
        ----------
        index_name : str
            The name of the OpenSearch index where documents will be stored.
        df : pandas.DataFrame
            The dataset containing processed text and associated metadata.
        model : Any
            A sentence embedding model supporting `.encode(text)` when
            embeddings need to be computed dynamically.
        embedding_by_index : dict or None
            Optional mapping from document IDs to precomputed embeddings. When
            provided, embeddings are retrieved from this map instead of being
            recomputed.

        Notes
        -----
        Uses `helpers.bulk` to efficiently upload all documents to OpenSearch.
        Progress is displayed using `tqdm` during DataFrame iteration.
        """
        def to_actions(df):
            for _, row in tqdm(df.iterrows(), total=df.shape[0]):
                text = row.preprocessed_text
                legislature = row.legislature
                legislative_session = row.legislative_session

                if embedding_by_index is not None:
                    embedding = self.get_embedding(embedding_by_index, row.doc_id)
                else:
                    embedding = model.encode(text)

                yield {
                    "_index": index_name,
                    "_id": row.doc_id,
                    "_source": {
                        "doc_id": row.doc_id,
                        "texto": text,
                        "legislature": legislature,
                        "legislative_session": legislative_session,
                        "embedding": embedding
                    }
                }

        helpers.bulk(self.client, to_actions(df))
        self.client.indices.refresh(index=index_name)

    def get_embedding(self, index_name, id):
        """
        Retrieves the embedding vector of a document stored in an OpenSearch index.

        The method performs a term query on the `doc_id` field to locate the
        corresponding document. If found, the embedding stored in the `_source`
        is returned; otherwise, an exception is raised.

        Parameters
        ----------
        index_name : str
            Name of the OpenSearch index to search.
        id : str or int
            The document identifier used to locate the stored embedding.

        Returns
        -------
        list or array
            The embedding vector associated with the document.

        Raises
        ------
        ValueError
            If no document with the given ID exists in the specified index.
        """
        query = {"query": {"term": {"doc_id": id}}}
        response = self.client.search(index=index_name, body=query)
        hits = response["hits"]["hits"]
        if hits:
            return hits[0]["_source"]["embedding"]
        else:
            raise ValueError(f"ID {id} not founded on index {index_name}.")

    def search_documment(self,
                         index_name,
                         embedding,
                         legislature=None,
                         legislative_session=None,
                         doc_id=None,
                         use_session=True
    ):
        """
        Performs a k-NN vector similarity search in an OpenSearch index using a query embedding.

        The method builds a `script_score` query that computes cosine similarity
        between the provided embedding and stored vectors. Optionally, results
        can be filtered by legislature and legislative session, and the document
        with the same `doc_id` can be excluded from the search results.

        Parameters
        ----------
        index_name : str
            Name of the OpenSearch index to query.
        embedding : list or array
            The embedding vector used as the query for similarity search.
        legislature : int, optional
            Legislative period used to filter results when `use_session` is True.
        legislative_session : int, optional
            Legislative session used to filter results when `use_session` is True.
        doc_id : str or int, optional
            Document ID to exclude from the result set.
        use_session : bool, default=True
            Whether to apply filtering by legislature and session.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the top retrieved documents with columns:
            `doc_id`, `texto`, and `score`.

        """


        body = {
            "size": 50,
            "_source": {"excludes": ["embedding"]},
            "query": {"script_score": {
                "query": {"bool": {"must_not": [{"term": {"doc_id": doc_id}}]}},
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {"field": "embedding", "query_value": embedding, "space_type": "cosinesimil"}
                }
            }}
        }

        if use_session:
            body["query"]["script_score"]["query"]["bool"]["must"] = [
                {"term": {"legislature": legislature}},
                {"term": {"legislative_session": legislative_session}}
            ]

        response = self.client.search(index=index_name, body=body)
        response = [
            [d["_source"]["doc_id"], d["_source"]["texto"], d["_score"]] 
            for d in response["hits"]["hits"]
        ]
        df_result = pd.DataFrame(response, columns=["doc_id", "texto", "score"])
        return df_result
