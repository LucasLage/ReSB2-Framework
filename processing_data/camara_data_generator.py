import re
import pandas as pd
import numpy as np
import random
import math

class CamaraDataGenerator:
    """
    A data generator class for handling Brazilian Chamber of Deputies (Câmara) proposition data.

    Attributes
    ----------
    data_path : str
        Path to the raw dataset containing the propositions.
    work_dir : str
        Base directory used for saving processed or intermediate files.
    seed : int
        Random seed for reproducibility in sampling and shuffling operations.
    df_original : pandas.DataFrame or None
        Will store the original loaded dataset.
    df_filtered : pandas.DataFrame or None
        Will store the filtered dataset after applying text or pattern-based filters.
    df_similars : pandas.DataFrame or None
        Will store identified similar proposition pairs.
    df_pairs_apensados : pandas.DataFrame or None
        Will store proposition pairs that are officially linked (apensadas).
    df_pairs_non_similars : pandas.DataFrame or None
        Will store generated non-similar proposition pairs.
    num_pares_nao_similnum_pairs_non_similars_completeares_completo : int or None
        Number of non-similar pairs to generate to balance the dataset.
    df_pairs_complete : pandas.DataFrame or None
        Will store the final combined dataset of similar and non-similar pairs.
    filter_pattern : str
        Text pattern used to filter propositions ('O Congresso Nacional decreta').
    """

    
    def __init__(self, data_path: str, work_dir: str, seed: int):
        """
        Initialize the Camara data generator with paths and configuration.

        Input:
        ------
        data_path : str
            Path to the raw Câmara dataset containing propositions.
        work_dir : str
            Directory used for saving processed data and outputs.
        seed : int
            Random seed for reproducibility of sampling and shuffling operations.

        """
        self.data_path = data_path
        self.work_dir = work_dir
        self.seed = seed
        self.df_original = None
        self.df_filtered = None
        self.df_similars = None
        self.df_pairs_apensados = None
        self.df_pairs_non_similars = None
        self.num_pairs_non_similars_complete = None
        self.df_pairs_complete = None

        self.filter_pattern = 'O Congresso Nacional decreta'

    def process(self):
        """
        Main processing pipeline for Câmara propositions.

        Steps performed:
        ----------------
        1. _load_data() : Load the raw proposition dataset.
        2. _filtrer() : Apply filtering to select relevant propositions based on patterns.
        3. _split_and_append_ementa() : Split and append proposition summaries (ementas).
        4. _process_apensados() : Process officially linked (apensadas) propositions.
        5. _generate_pairs_apensados() : Generate labeled similar pairs from linked propositions.
        6. _generate_pairs_non_similars() : Generate labeled non-similar pairs for training/testing.
        7. _save_results() : Save the generated datasets to disk.
        8. _gerar_data_mlm() : Generate remaining documents for Masked Language Modeling (MLM) task.
        """
        self._load_data()
        self._filtrer()
        self._split_and_append_ementa()
        self._process_apensados()
        self._generate_pairs_apensados()
        self._generate_pairs_non_similars()
        self._save_results()
        self._gerar_data_mlm()


    def _load_data(self):
        """
        Load the raw Câmara dataset into a DataFrame.

        Output:
        -------
        df_original : pandas.DataFrame
            Contains the loaded raw Câmara propositions.
        """
        self.df_original = pd.read_csv(self.data_path)
        self.df_original['txtSiglaTipo'] = self.df_original['txtSiglaTipo'].str.strip()
        print(f"Camara - Data loaded: {self.df_original.shape[0]} row")


    def _filtrer(self):
        """
        Apply filtering and preprocessing to the raw Câmara dataset.

        Output:
        -------
        df_filtered : pandas.DataFrame
            Cleaned and filtered dataset ready for subsequent processing steps.
        """
        df = self.df_original.copy()
        df['txtInteiroTeorOriginal'] = df['txtInteiroTeor']
        df = df[df['txtInteiroTeor'].str.len() >= 100]

        # Remove justification sections from the text
        df['txtInteiroTeor'] = df['txtInteiroTeor'].apply(
            lambda text: re.split(r'Justifica[çc][aã]o|Justificativa|J U S T I F I C A T I V A|JUSTIFICATIVA|JUSTIFICAÇÃO|EXPOSIÇÃO DE MOTIVOS|Exposição de motivos', text)[0] if isinstance(text, str) else text
        )

        # Keep only documents that contain the required legislative pattern
        df = df[df['txtInteiroTeor'].str.contains(self.filter_pattern, na=False, case=False)]
        df.drop_duplicates(subset=["txtNome"], keep='first', inplace=True)
        self.df_filtered = df
        print(f"Camara - Document filtered: {df.shape[0]} rows")


    def _split_and_append_ementa(self):
        """
        Extracts the main body of each document starting from the legislative trigger 
        phrase (defined in `self.filter_pattern`) and prepends the document's ementa 
        (summary). This step standardizes the structure of the text used in later 
        pair-generation and modeling tasks.
        
        The final structure becomes:
        
            <Ementa>
            
            <"O Congresso Nacional decreta..." + remaining content>
        
        The operation is applied row-by-row to the filtered DataFrame.
        """
        def aplicar(row):
            texto_completo = row['txtInteiroTeor']
            ementa = row['txtEmenta']
            texto_completo_lower = texto_completo.lower()

            lower_pattern = self.filter_pattern.lower()
            start_index = texto_completo_lower.find(lower_pattern)
            if start_index != -1:
                parte_depois_raw = texto_completo[start_index:]
                parte_depois = parte_depois_raw.strip()
                if not parte_depois.startswith(lower_pattern):
                    parte_depois = lower_pattern + '\n\n' + parte_depois.replace(lower_pattern, '', 1).strip()
                else:
                    parte_depois = lower_pattern + '\n\n' + parte_depois.replace(lower_pattern, '', 1).strip()
            else:
                print(texto_completo_lower, row)
            novo_texto = ementa.strip() + '\n\n' + parte_depois.strip()
        
            return novo_texto

        self.df_filtered['txtInteiroTeor'] = self.df_filtered.apply(aplicar, axis=1)


    @staticmethod
    def _check_apensados(txtNome, arvore):
        """
        Recursively searches an 'apensados' (attached propositions) tree to locate
        the subtree associated with the given proposition name (`txtNome`).

        Parameters
        ----------
        txtNome : str
            The name of the proposition to search for within the attachment tree.
        arvore : dict
            A nested dictionary representing the hierarchical structure of
            propositions and their attached (apensados) children. Expected keys:
            - 'nomeProposicao': str
            - 'apensados': list of nested dicts in the same format

        Returns
        -------
        list or None
            The list of attached propositions (subtree) corresponding to `txtNome`
            if found. If the proposition does not appear anywhere in the tree,
            the function returns None.

        Notes
        -----
        This method performs a depth-first recursive search. It checks the current
        node first; if it does not match, it recursively explores each child in 
        the 'apensados' list until a match is found or the tree is fully traversed.
        """
        if arvore['nomeProposicao'] == txtNome:
            return arvore['apensados']
        for a in arvore['apensados']:
            resultado = CamaraDataGenerator._check_apensados(txtNome, a)
            if resultado is not None:
                return resultado
        return None

    @staticmethod
    def _apensados_all_levels(arvore: list, propositions_dataset: set) -> list:
        """
        Recursively extract all valid 'apensados' (attached propositions) from every
        level of the attachment tree.

        Input:
        arvore : list
            A list representing the current attachment subtree, where each element
            contains 'nomeProposicao' and a nested 'apensados' list.
        propositions_dataset : set
            Set of proposition identifiers considered valid for filtering.

        Returns:
        list
            A flat list containing all proposition names found across the entire
            recursive tree structure that also exist in the dataset.
        """
        if not arvore:
            return []
        filhos = []
        for filho in arvore:
            if filho['nomeProposicao'] in propositions_dataset:
                filhos.append(filho['nomeProposicao'])
            filhos.extend(CamaraDataGenerator._apensados_all_levels(filho['apensados'], propositions_dataset))
        return filhos

    def _process_apensados(self):
        """
        Process and extract all 'apensados' (attached propositions) for filtered documents.

        Steps:
        - Identify rows that contain a valid JSON tree of attachments ('jsonArvoreApensacao').
        - For each proposition, retrieve its corresponding attachment subtree.
        - Recursively extract all attached propositions across all hierarchical levels.
        - Keep only propositions that actually have at least one valid attachment.
        - Store the resulting DataFrame in `self.df_similars`.

        Output:
        - Updates `self.df_similars` with documents that have at least one extracted
          attachment at any level.
        """
        df_sim = self.df_filtered[self.df_filtered['jsonArvoreApensacao'].apply(
            lambda arvore: isinstance(arvore, str) and arvore is not None and len(arvore) > 0
        )].copy()

        df_sim['arvore_individual'] = df_sim.apply(
            lambda row: self._check_apensados(row['txtNome'], eval(row['jsonArvoreApensacao'])),
            axis=1
        )

        propositions_dataset = set(df_sim['txtNome'])

        df_sim['apensados_todos_os_niveis'] = df_sim.apply(
            lambda row: self._apensados_all_levels(row['arvore_individual'], propositions_dataset),
            axis=1
        )

        df_sim = df_sim[df_sim['apensados_todos_os_niveis'].apply(lambda x: len(x) > 0)]
        self.df_similars = df_sim

    def _generate_pairs_apensados(self):
        """
        Generate a DataFrame containing all positive pairs derived from
        'apensados' (attached propositions).

        This method converts the hierarchical attachment relationships into
        pairwise training data, where each proposition is paired with every
        other proposition found in its full attachment tree.

        Returns:
        None
            The resulting DataFrame is stored in `self.df_pairs_apensados`.
        """

        id_para_texto = self.df_filtered.set_index('txtNome')['txtInteiroTeor'].to_dict()
        pares_apensados = []
    
        for _, row in self.df_similars.iterrows():
            id1 = row['txtNome']
            txt1 = row['txtInteiroTeor']
            apensados = row['apensados_todos_os_niveis']
            for id2 in apensados:
                txt2 = id_para_texto.get(id2)
                pares_apensados.append({
                    'doc_id_1': id1,
                    'preprocessed_text_1': txt1,
                    'doc_id_2': id2,
                    'preprocessed_text_2': txt2,
                    'label': 1
                })
    
        self.df_pairs_apensados = pd.DataFrame(pares_apensados)
            
        num_pairs_similars_complete = len(self.df_pairs_apensados)

        # 3:1 negative:positive ratio. 75% non similar 25% similar
        self.num_pairs_non_similars_complete = int(num_pairs_similars_complete * 3)
        print(f"Camara - Pairs similars: {self.df_pairs_apensados.shape[0]} rows")


    def _generate_pairs_non_similars(self):
        """
        Generate non-similar pairs.

        Separate documents into those that are part of similar pairs and those that are not.
        similar – non_similar pairs: one document is in the similar set, the other is not.
        non_similar – non_similar pairs: neither document is in the similar set.

        In both cases, documents are sampled randomly.

        Output:

        df_pairs_non_similars : pandas.DataFrame
        """


        rng = random.Random(self.seed)
        pairs_set = set()
        pairs_non_similars = []

        
        ids_similars = (
            self.df_pairs_apensados["doc_id_1"].tolist() +
            self.df_pairs_apensados["doc_id_2"].tolist()
        )
        ids_similars = list(set(ids_similars))
        df_similars = self.df_filtered[self.df_filtered["txtNome"].isin(ids_similars)]
        df_similars = df_similars.sample(frac=1, random_state=self.seed)
        
        print(f"Camara - Number of distinct documents among similar pairs: : {len(df_similars)}")

        df_non_similars = self.df_filtered[~self.df_filtered["txtNome"].isin(ids_similars)]
        print(f"Camara - Number of distinct documents outside the similar pairs: {len(df_non_similars)}")

        df_non_similars = df_non_similars.sample(frac=1, random_state=self.seed)

        # Use only one-third of the total data to ensure sufficient remaining data for the MLM task.
        top = len(df_non_similars) - (len(df_non_similars) // 3)

        df_non_similars = df_non_similars.head(top)
        print(f"Camara - Number of distinct documents outside the similar pairs for sampling: {len(df_non_similars)}")


        ####### Generating random similar–non-similar pairs
        # Randomly pair one similar document with one non-similar document
        pairs = zip(df_similars.iterrows(), df_non_similars.iterrows())
        for (_, row_similar), (_, row_non_similar) in pairs:
            # Add the pair to the set of already generated pairs
            pair_key = tuple(sorted((row_similar.txtNome, row_non_similar.txtNome)))
            pairs_set.add(pair_key)

            pairs_non_similars.append(
                {
                    "doc_id_1": row_similar.txtNome,
                    "preprocessed_text_1": row_similar.txtInteiroTeor,
                    "doc_id_2": row_non_similar.txtNome,
                    "preprocessed_text_2": row_non_similar.txtInteiroTeor,
                    "label": 0
                }
            )            

        ids_non_similars = df_non_similars.txtNome.tolist()
        n_docs = len(ids_non_similars)
        df_query_text = df_non_similars.copy().set_index("txtNome")
        
        ####### Generating random non-similar–non-similar pairs
        while len(pairs_non_similars) < self.num_pairs_non_similars_complete:
            # Select two random indices
            i1, i2 = rng.sample(range(n_docs), 2)

            id1, id2 = ids_non_similars[i1], ids_non_similars[i2]
            pair_key = tuple(sorted((id1, id2)))

            if pair_key in pairs_set:
                continue

            # Add the pair to the set of already generated pairs
            pairs_set.add(pair_key)

            # Retrieve texts
            txt1 = df_query_text.loc[id1]["txtInteiroTeor"]
            txt2 = df_query_text.loc[id2]["txtInteiroTeor"]

            pairs_non_similars.append(
                {
                    "doc_id_1": id1,
                    "preprocessed_text_1": txt1,
                    "doc_id_2": id2,
                    "preprocessed_text_2": txt2,
                    "label": 0
                }
            )
        self.df_pairs_non_similars = pd.DataFrame(pairs_non_similars)
        df = self.df_pairs_non_similars
        
        len_ns = len(df[(df.label == 0)])
        
        print(f"Camara - Total number of non-similar pairs: {len_ns}")
    

    def _save_results(self):
        """
        Combine similar and non-similar pairs, remove duplicates, and save to CSV.

        Input:
        ------
        df_pairs_apensados : pandas.DataFrame
            DataFrame containing similar proposition pairs.
        df_pairs_non_similars : pandas.DataFrame
            DataFrame containing non-similar proposition pairs.

        Output:
        -------
        df_pairs_complete : pandas.DataFrame
            Combined DataFrame of labeled pairs after removing duplicates.
        """
        self.df_pairs_complete = pd.concat([self.df_pairs_apensados, self.df_pairs_non_similars], ignore_index=True)
        len_completo = len(self.df_pairs_complete)
        self.df_pairs_complete = self.df_pairs_complete[
            self.df_pairs_complete.preprocessed_text_1 != self.df_pairs_complete.preprocessed_text_2 
        ]
        print(f"Camara - Number of pairs removed due to duplication {len_completo - len(self.df_pairs_complete)}")
        
        output_file = f"{self.work_dir}/data/sb2_chamber_pairs.csv"
        self.df_pairs_complete.to_csv(output_file, index=False)
        print(f"Camara- File saved {output_file}")

    def _gerar_data_mlm(self):
        """
        Generate a dataset for Masked Language Modeling (MLM) using documents
        that were not included in the labeled similar/non-similar pairs.

        Input:
        ------
        df_pairs_complete : pandas.DataFrame
            DataFrame containing labeled pairs of similar and non-similar propositions.

        Output:
        -------
        Saves a CSV file containing the remaining documents for MLM training.
        """

        all_docs = set(self.df_pairs_complete["doc_id_1"]) | set(self.df_pairs_complete["doc_id_2"])
        df_mlm = self.df_filtered[~self.df_filtered["txtNome"].isin(all_docs)]

        df_mlm = df_mlm.rename(columns={
            "txtNome": "doc_id",
            "txtInteiroTeor": "preprocessed_text"
        })[["doc_id", "preprocessed_text"]]
        df_mlm.to_csv(f"{self.work_dir}/data/sb2_chamber_mlm.csv", index=False)
        print(f"Camara - Data MLM saved ({df_mlm.shape[0]} rows)")
