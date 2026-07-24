import pandas as pd
import numpy as np
import random
import json
import re
import math

class AlmgDataGenerator:
    """
    A data generator class for creating similarity and MLM datasets from ALMG legislative propositions.

    Attributes
    ----------
    work_dir : str
        Base directory where outputs (datasets, preprocessed files) will be stored.
    seed : int
        Random seed for reproducibility of data processing and sampling.
    df_proposition : pandas.DataFrame or None
        Will store the loaded proposition data, initialized as None until `_load_data` is called.
    """


    def __init__(self, work_dir: str, seed: int):
        """
        Initialize the ALMG data generator with the working directory and random seed.

        Input:

        work_dir : str
            Base directory where all output datasets and files will be stored.
        seed : int
            Random seed to ensure reproducibility in sampling and processing.
        """
        self.work_dir = work_dir
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.df_proposition = None


    def process(self):
        """
        Main processing function to generate a dataset for similarity tasks.

        Steps:
        1. Load the raw proposition data using `_load_data`.
        2. Generate pairs of similar propositions using `_get_similars`.
        3. Generate pairs of non-similar propositions using `_get_non_similars`.
        4. Combine similar and non-similar pairs, label them, and save to disk using `_combine_and_save`.
        5. Extract additional similar pairs from existing projects using `_get_similars_ex_project`.
        6. Generate MLM (Masked Language Modeling) data from the labeled dataset using `_generate_data_mlm`.
        """
        self._load_data()
        df_similar = self._get_similars()
        df_not_similar = self._get_non_similars(df_similar)
        df_labeled = self._combine_and_save(df_similar, df_not_similar)
        self._get_similars_ex_project()
        self._generate_data_mlm(df_labeled)


    def _load_data(self):
        """
        Load the preprocessed ALMG proposition data.
        """
        path = f"{self.work_dir}/data/almg_proposition_pre_processed.csv"
        self.df_proposition = pd.read_csv(path)
        print(f"Data loaded: {self.df_proposition.shape[0]} rows")


    def _get_similars(self):
        """
        Identify similar propositions based on anexadas/anexadoras relationships and former projects.

        Output:

        df_similar : pandas.DataFrame
        """
        print("Generating ALMG similar pairs...")
        df = self.df_proposition
        similars = []

        similars += self._get_similars_anexadas(df)
        similars += self._get_similars_anexadoras(df)

        similars = [list(x) for x in set(tuple(s) for s in similars)]

        df_similar = pd.DataFrame(
            similars,
            columns=["doc_id_1", "preprocessed_text_1", "doc_id_2", "preprocessed_text_2", "label"],
        )

        print(f"Similars founded: {len(df_similar)}")
        return df_similar

    def _get_similars_anexadas(self, df):
        """
        Generate pairs of similar propositions where the first proposition has anexadas documents.

        Input:

        df : pandas.DataFrame
            DataFrame containing all propositions with their metadata and preprocessed text.

        Output:

        similars : list of lists
        """
        similars = []
        subset = df[df["linksProposicoesAnexadas"].notna()]
        for _, row in subset.iterrows():
            anexadas = eval(row["linksProposicoesAnexadas"])
            if isinstance(anexadas, dict):
                anexadas = [anexadas]
            for a in anexadas:
                tipo, numero, ano = AlmgDataGenerator._parse_description(a["descricao"])
                df_it = df[
                    (df["siglaTipoProjeto"] == tipo)
                    & (df["numero"] == numero)
                    & (df["ano"] == ano)
                ]
                for _, row_it in df_it.iterrows():
                    similars.append([
                        row["doc_id"],
                        row["preprocessed_text"],
                        row_it["doc_id"],
                        row_it["preprocessed_text"],
                        1
                    ])
        return similars

    def _get_similars_anexadoras(self, df):
        """
        Generate pairs of similar propositions where the first proposition has anexadoras documents.

        Input:

        df : pandas.DataFrame
            DataFrame containing all propositions with their metadata and preprocessed text.

        Output:

        similars : list of lists
        """
        similars = []
        subset = df[df["linkProposicaoAnexadora"].notna()]
        for _, row in subset.iterrows():
            anexadoras = eval(row["linkProposicaoAnexadora"])
            if isinstance(anexadoras, dict):
                anexadoras = [anexadoras]
            for a in anexadoras:
                tipo, numero, ano = AlmgDataGenerator._parse_description(a["descricao"])
                df_it = df[
                    (df["siglaTipoProjeto"] == tipo)
                    & (df["numero"] == numero)
                    & (df["ano"] == ano)
                ]
                for _, row_it in df_it.iterrows():
                    similars.append([
                        row_it["doc_id"],
                        row_it["preprocessed_text"],
                        row["doc_id"],
                        row["preprocessed_text"],
                        1
                    ])
        return similars

    def _get_similars_ex_project(self):
        """
        Detect cases of 'Ex-Projeto de Lei nº X/XXXX' within proposition texts
        and generate pairs of similar propositions based on these references.

        These ex-project pairs will be used during the test stage to reduce false negatives in the dataset.

        Output:

        A CSV file containing pairs of propositions where one references
        a former project.
        """
        print("Searching Ex-Projetos...")
        
        df = self.df_proposition
        padrao = r'Ex[-\s]?Projeto de Lei\s*(?:n[º°]?\s*)?([\d\.]+)\s*/\s*(\d{4}|\d{2})'

        def contem_padrao(texto):
            return bool(re.search(padrao, texto, flags=re.IGNORECASE))

        def extrair_num_ano(texto):
            match = re.findall(padrao, texto, flags=re.IGNORECASE)[0]
            num = int(match[0].replace(".", ""))
            ano_str = match[1]
            ano = int(ano_str)
            if len(ano_str) == 2:
                ano = 1900 + ano if ano >= 30 else 2000 + ano
            return num, ano

        df_ex = df[df["preprocessed_text_num"].apply(contem_padrao)]
        new_sim = []
        for _, row in df_ex.iterrows():
            num, ano = extrair_num_ano(row["preprocessed_text_num"])
            df_match = df[
                (df["numero"] == num)
                & (df["ano"] == ano)
                & (df["siglaTipoProjeto"] == "PL")
            ]
            if not df_match.empty:
                row2 = df_match.iloc[0]
                new_sim.append([
                    row["doc_id"],
                    row["preprocessed_text"],
                    row2["doc_id"],
                    row2["preprocessed_text"],
                    1
                ])
        df_similars_ex_project = pd.DataFrame(new_sim, columns=["doc_id_1", "preprocessed_text_1",
                                              "doc_id_2", "preprocessed_text_2", "label"])

        output = f"{self.work_dir}/data/almg_similars_ex_project.csv"
        df_similars_ex_project.to_csv(output, index=False)
        print(f"Similars Ex project saved {output}")


    def _get_non_similars(self, df_similar):
        """
        Generate non-similar pairs.

        Separate documents into those that are part of similar pairs and those that are not.
        similar – non_similar pairs: one document is in the similar set, the other is not.
        non_similar – non_similar pairs: neither document is in the similar set.

        In both cases, documents are sampled randomly.
        
        Input:

        df_similar : pandas.DataFrame
            DataFrame containing pairs of similar propositions with their IDs and preprocessed text.

        Output:

        df_not_similar : pandas.DataFrame
        """
        
        print("Generating ALMG non similar pairs...")

        rng = random.Random(self.seed)
        pairs_set = set()
        pairs_non_similares = []

        # 3:1 negative:positive ratio. 75% non similar 25% similar
        num_pairs_non_similars_complete = len(df_similar) * 3
        
        ids_similars = (
            df_similar["doc_id_1"].tolist() +
            df_similar["doc_id_2"].tolist()
        )
        ids_similars = list(set(ids_similars))
        df_similars = self.df_proposition[self.df_proposition["doc_id"].isin(ids_similars)]
        df_similars = df_similars.sample(frac=1, random_state=self.seed)

        print(f"ALMG - Number of distinct documents among similar pairs: {len(df_similars)}")

        df_non_similars = self.df_proposition[~self.df_proposition["doc_id"].isin(ids_similars)]
        print(f"ALMG - Number of distinct documents outside the similar pairs: {len(df_non_similars)}")
        
        df_non_similars = df_non_similars.sample(frac=1, random_state=self.seed)

        # Use only one-third of the total data to ensure sufficient remaining data for the MLM task.
        top = len(df_non_similars) - (len(df_non_similars) // 3)

        df_non_similars = df_non_similars.head(top)
        print(f"ALMG - Number of distinct documents outside the similar pairs for sampling: {len(df_non_similars)}")

        ####### Generating random similar–non-similar pairs
        # Randomly pair one similar document with one non-similar document
        pairs = zip(df_similars.iterrows(), df_non_similars.iterrows())
        for (_, row_similar), (_, row_non_similar) in pairs:
            # Add the pair to the set of already generated pairs
            pair_key = tuple(sorted((row_similar.doc_id, row_non_similar.doc_id)))
            pairs_set.add(pair_key)

            pairs_non_similares.append(
                {
                    "doc_id_1": row_similar.doc_id,
                    "preprocessed_text_1": row_similar.preprocessed_text,
                    "doc_id_2": row_non_similar.doc_id,
                    "preprocessed_text_2": row_non_similar.preprocessed_text,
                    "label": 0
                }
            )
        ids_non_similars = df_non_similars.doc_id.tolist()
        n_docs = len(ids_non_similars)
        df_query_text = df_non_similars.copy().set_index("doc_id")
        
        ####### Generating random non-similar–non-similar pairs
        while len(pairs_non_similares) < num_pairs_non_similars_complete:
            # Select two random indices
            i1, i2 = rng.sample(range(n_docs), 2)

            id1, id2 = ids_non_similars[i1], ids_non_similars[i2]
            pair_key = tuple(sorted((id1, id2)))

            if pair_key in pairs_set:
                continue

            # Add the pair to the set of already generated pairs
            pairs_set.add(pair_key)

            # Retrieve texts
            txt1 = df_query_text.loc[id1]["preprocessed_text"]
            txt2 = df_query_text.loc[id2]["preprocessed_text"]

            pairs_non_similares.append(
                {
                    "doc_id_1": id1,
                    "preprocessed_text_1": txt1,
                    "doc_id_2": id2,
                    "preprocessed_text_2": txt2,
                    "label": 0
                }
            )

        df_not_similar_data = pd.DataFrame(pairs_non_similares)
        df = df_not_similar_data
        
        len_ns = len(df[(df.label == 0)])
        
        print(f"ALMG - Total number of non-similar pairs: {len_ns}")

        return df_not_similar_data

    

    def _combine_and_save(self, df_similar, df_not_similar):
        """
        Combine similar and non-similar pairs, remove duplicates, and save to CSV.

        Input:
        ------
        df_similar : pandas.DataFrame
            DataFrame containing similar proposition pairs.
        df_not_similar : pandas.DataFrame
            DataFrame containing non-similar proposition pairs.

        Output:
        -------
        df_labeled : pandas.DataFrame
            Combined DataFrame of labeled pairs after removing duplicates.
        """
        df_labeled = pd.concat([df_similar, df_not_similar], ignore_index=True)
        len_complete = len(df_labeled)
        df_labeled = df_labeled[
            df_labeled.preprocessed_text_1 != df_labeled.preprocessed_text_2 
        ]
        print(f"ALMG - Number of pairs removed due to duplication {len_complete - len(df_labeled)}")
        
        
        output = f"{self.work_dir}/data/sb2_almg_pairs.csv"
        df_labeled.to_csv(output, index=False)
        print(f"ALMG - File saved {output}")
        return df_labeled

    def _generate_data_mlm(self, df_labeled):
        """
        Generate a dataset for Masked Language Modeling (MLM) using documents
        that were not included in the labeled similar/non-similar pairs.

        Input:
        ------
        df_labeled : pandas.DataFrame
            DataFrame containing labeled pairs of similar and non-similar propositions.

        Output:
        -------
        Saves a CSV file containing the remaining documents for MLM training.
        """
        all_docs = set(df_labeled["doc_id_1"]) | set(df_labeled["doc_id_2"])
        df_mlm = self.df_proposition[~self.df_proposition["doc_id"].isin(all_docs)]
        df_mlm = df_mlm[["doc_id", "preprocessed_text"]]
        df_mlm.to_csv(f"{self.work_dir}/data/sb2_almg_mlm.csv", index=False)
        print(f"ALMG - Data MLM saved ({df_mlm.shape[0]} rows)")


    @staticmethod
    def _parse_description(desc):
        """
        Parse a proposition description string into its components.

        Input:
        ------
        desc : str
            Description string in the format "TYPE NUMBER YEAR" (e.g., "PL 1234 2021").

        Output:
        ------
        tipo : str
            Proposition type (e.g., "PL").
        numero : int
            Proposition number.
        ano : int
            Year of the proposition.
        """
        values = desc.split(" ")
        tipo = values[0].replace(".", "")
        numero = int(values[1])
        ano = int(values[2])
        return tipo, numero, ano
