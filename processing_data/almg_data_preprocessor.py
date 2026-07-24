import pandas as pd
import numpy as np
import re
from html import unescape
import unicodedata



class AlmgDataPreprocessor:
    """
    A preprocessor class for handling ALMG legislative proposition data.

    Attributes
    ----------
    proposition_detail_path : str
        Path to the CSV/JSON file containing detailed proposition metadata.
    proposition_text_path : str
        Path to the file containing the full text of propositions.
    legislative_sessions_path : str
        Path to the dataset containing legislative session information.
    work_dir : str
        Base directory used for saving intermediate or processed files.
    df_proposition_detail : pandas.DataFrame or None
        Will store the loaded proposition metadata after initialization.
    df_text : pandas.DataFrame or None
        Will store the loaded proposition text data.
    df_sessions : pandas.DataFrame or None
        Will store the loaded legislative session information.
    """

    def __init__(
            self,
            proposition_detail_path: str,
            proposition_text_path: str,   
            legislative_sessions_path: str,
            work_dir: str
    ):
        """
        Initialize the ALMG data preprocessor with paths to the required datasets.

        Input:

        proposition_detail_path : str
            Path to the proposition metadata file.
        proposition_text_path : str
            Path to the file containing full proposition texts.
        legislative_sessions_path : str
            Path to the legislative sessions dataset.
        work_dir : str
            Directory used for saving processed data outputs.
        """
        self.proposition_detail_path = proposition_detail_path
        self.proposition_text_path = proposition_text_path
        self.legislative_sessions_path = legislative_sessions_path
        self.work_dir = work_dir
        self.df_proposition_detail = None
        self.df_text = None
        self.df_sessions = None

    def process(self):
        """
        Run the full ALMG data preprocessing pipeline.

        Steps
        -----
        1. _load_data()
            Loads raw proposition metadata, proposition text, and legislative session files
            into internal DataFrames.

        2. _merge_data()
            Merges the loaded datasets into a unified structure, ensuring each proposition
            contains text, metadata, and session information when available.

        3. _text_preprocessing()
            Applies text normalization procedures such as lowercasing, removing extra spaces,
            fixing encoding issues, and general cleaning needed for NLP models.

        4. _remove_justification()
            Identifies and removes "justification" sections from proposition texts,
            which are often long, noisy, and not useful for similarity tasks.

        5. _remove_law_numbers()
            Removes explicit law numbers and legal references to reduce noise and prevent
            models from overfitting on specific legal identifiers.

        6. _infer_legislative_session()
            Infers the corresponding legislative session for each proposition using
            metadata rules and the sessions dataset.

        7. _save()
            Saves the final processed dataset to the working directory in a clean,
            analysis-ready format.
        """
        self._load_data()
        self._merge_data()
        self._text_preprocessing()
        self._remove_justification()
        self._remove_law_numbers()
        self._infer_legislative_session()
        self._save()


    def _load_data(self):
        """
        Loads proposition details, text data, and legislative sessions.
        """
        print(f"ALMG - Loading data.")
        self.df_proposition_detail = pd.read_json(self.proposition_detail_path, lines=True)
        
        self.df_text = pd.read_json(self.proposition_text_path, lines=True)

        self.df_sessions = pd.read_json(self.legislative_sessions_path, lines=True)
        
        self.df_sessions['datetime_inicio'] = pd.to_datetime(
            self.df_sessions['adat_inicio'], format='%d/%m/%Y %H:%M:%S', errors='coerce'
        )
        self.df_sessions['datetime_termino'] = pd.to_datetime(
            self.df_sessions['adat_termino'], format='%d/%m/%Y %H:%M:%S', errors='coerce'
        )


    def _merge_data(self):
        """
        Merges proposition metadata with text, keeps only initial texts,
        cleans empty entries, fixes publication dates, and assigns doc_id.
        """
        print(f"ALMG - Merging data.")

        # Create a cod_text column to filter only the initial text of each proposition
        self.df_text['cod_text'] = self.df_text['codigo'].apply(lambda x: x[-2:])
        self.df_text = self.df_text.rename(columns={"dataPublicacao": "dataPublicacao_txt"})

        self.df_proposition_detail = pd.merge(
            self.df_proposition_detail,
            self.df_text,
            on=['siglaTipoProjeto', 'numero', 'ano', 'complemento'],
            how='left'
        )
        
        self.df_proposition_detail.dropna(subset=['texto'])
        self.df_proposition_detail['texto'] = self.df_proposition_detail['texto'].str.strip()
        self.df_proposition_detail = self.df_proposition_detail[self.df_proposition_detail['texto'] != '']
        
        # Filters only the initial text of each proposition
        self.df_proposition_detail = self.df_proposition_detail[self.df_proposition_detail['cod_text'] == '01']

        self.df_proposition_detail["data_publicacao"] = self.df_proposition_detail["dataPublicacao"].fillna(
            self.df_proposition_detail["dataPublicacao_txt"]
        )

        self.df_proposition_detail['doc_id'] = self.df_proposition_detail.index
        self.df_proposition_detail = self.df_proposition_detail[[
            'doc_id', 'siglaTipoProjeto', 'numero', 'ano', 'complemento', 'texto', 'cod_text',
            'linksProposicoesAnexadas', 'linkProposicaoAnexadora', 'listaIndexacao', 'data_publicacao'
        ]]


    @staticmethod
    def _remove_css(t: str) -> str:
        """
        Removes CSS-like blocks from a text string.

        Input:
            t (str): Raw text that may contain CSS formatting blocks

        Output:
            str: The cleaned text with CSS-style blocks removed.
        """
        text = t.replace("@page ", "")
    
        stack = []
        pairs = []
        
        for i, character in enumerate(text):
            if character == '{':
                stack.append(i)  # Adds the position of the opening brace to the stack
            elif character == '}':
                if stack:
                    open_pos = stack.pop()  # Removes the position of the most recent opening brace from the stack
                    pairs.append((open_pos, i))  # Adds the pair of positions to the list of pairs
    
        if len(pairs) == 0:
            return text
        
        begin = pairs[0][0]
        end = pairs[0][1]
        
        for i in reversed(range(1, len(pairs))):
            p1 = pairs[i]
            p2 = pairs[i-1]
            text_aux = text[:p1[0]] + text[p1[1]:]
            if (":" in text_aux) and (";" in text_aux) and ((p1[0] - p2[1]) < 20):
                end = p1[1]
                break
        
        text = text[:begin] + text[end+1:]
        return text

    @staticmethod
    def _pre_process_txt(texto: str) -> str:
        """
        Preprocesses a text string by performing multiple cleaning steps.

        Input:
            texto (str): Raw text that may contain URLs, CSS blocks, HTML entities,
                         and unwanted unicode/control characters.

        Output:
            str: Cleaned text.

        """
        texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
        texto = re.sub(r'http?://\S+|www\.\S+', '', texto)
        texto = unescape(texto)
        texto = AlmgDataPreprocessor._remove_css(texto)
        texto = texto.strip()
        texto = ''.join(c for c in texto if unicodedata.category(c)[0] not in ['C', 'Z'] or c == ' ')
        return texto

    def _text_preprocessing(self):
        """
        Preprocesses a text string by performing multiple cleaning steps.

        Input:
            texto (str): Raw text that may contain URLs, CSS blocks, HTML entities,
                         and unwanted unicode/control characters.

        Output:
            str: Cleaned text

        """
        print(f"ALMG - Pre processing data.")
        self.df_proposition_detail['preprocessed_text'] = self.df_proposition_detail['texto'].apply(
            lambda x: AlmgDataPreprocessor._pre_process_txt(x)
        )
        self.df_proposition_detail['preprocessed_text_num'] = self.df_proposition_detail['preprocessed_text']

    def _remove_justification(self):
        """
        Removes the 'justification' section from each proposition's preprocessed text.
        """
        pattern = r'(?i)justifica[çc][aã]o\s*:'

        def remove(row):
            return re.split(pattern, row['preprocessed_text'], maxsplit=1)[0]

        self.df_proposition_detail['preprocessed_text'] = self.df_proposition_detail.apply(remove, axis=1)

    def _remove_law_numbers(self):
        """
        Removes explicit law numbers from the preprocessed text of each proposition.
        """
        pattern = r"\d+(?:\.\d+)?/\d{2,4}"
        self.df_proposition_detail["preprocessed_text"] = (
            self.df_proposition_detail["preprocessed_text"]
            .str.replace(pattern, "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    @staticmethod
    def _get_legislative_session(df_sessions, datetime_val):
        """
        Determines the legislative session corresponding to a given datetime.

        This method is used to assign a proposition to the correct legislative session
        based on its publication date or any other relevant timestamp.

        Input:
            df_sessions (pandas.DataFrame): DataFrame containing legislative sessions.
            datetime_val (datetime): The datetime for which the legislative session
                                     should be determined.

        Output:
            tuple: (anum_legislatura, anum_sessao_legislativa) if a session is found,
                   or (None, None) if no session matches.

        """
        df_aux = df_sessions[
            (datetime_val >= df_sessions['datetime_inicio'])
            &
            (
                (df_sessions['datetime_termino'].isna())
                |
                (datetime_val <= df_sessions['datetime_termino'])
            )
        ]
        # If a current session is found → return
        if not df_aux.empty:
            row = df_aux.iloc[0]
            return row.anum_legislatura, row.anum_sessao_legislativa
    
        # If none is found, look for the next future session
        df_future = df_sessions[
            df_sessions["datetime_inicio"] > datetime_val
        ].sort_values("datetime_inicio")
    
        if not df_future.empty:
            row = df_future.iloc[0]
            return row.anum_legislatura, row.anum_sessao_legislativa
    
        # If no future session exists either, return None
        return None, None


    def _infer_legislative_session(self):
        """
        Infers the legislative session for each proposition based on its publication date.

        This method adds two new columns to the proposition DataFrame:
        - 'legislature': the legislature number corresponding to the proposition's date
        - 'legislative_session': the session number corresponding to the proposition's date

        """
        self.df_proposition_detail['datetime_publicacao'] = pd.to_datetime(
            self.df_proposition_detail['data_publicacao'], format='%Y%m%d', errors='coerce'
        )
        self.df_proposition_detail[['legislature', 'legislative_session']] = self.df_proposition_detail['datetime_publicacao'].apply(
            lambda x: pd.Series(AlmgDataPreprocessor._get_legislative_session(self.df_sessions, x))
        )

    def _save(self):
        """
        Saves the preprocessed proposition data to a CSV file.
        """
        output_file = f"{self.work_dir}/data/almg_proposition_pre_processed.csv"
        self.df_proposition_detail.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")

