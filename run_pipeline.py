# python3 -m venv pipeline_venv
# source pipeline_venv/bin/activate
# nohup python3 run_pipeline.py > ../log.txt 2>&1 &


import subprocess

import argparse
import sys, os
from datetime import datetime
from huggingface_hub import snapshot_download
from processing_data.camara_data_generator import CamaraDataGenerator
from processing_data.almg_data_preprocessor import AlmgDataPreprocessor
from processing_data.almg_data_generator import AlmgDataGenerator
from processing_data.split_save_data import split_save_data

from testing.model_tester import ModelTester
from testing.open_search_operator import OpenSearchOperator


def run(
    camara_data_path,
    proposition_detail_path,
    proposition_text_path,
    legislative_sessions_path,
    base_model,
    contex_size,
    open_ai_index,
    os_host,
    os_port,
    os_user,
    os_password,
    seed
):
    # Build the output directory
    timestamp = datetime.now().strftime("%Y_%m_%d")
    work_dir = f"../output_{timestamp}"
    os.makedirs(work_dir, exist_ok=True)

    # Build the data output directory
    data_path = f"{work_dir}/data"
    os.makedirs(data_path, exist_ok=True)

    # Generate Câmara data
    CamaraDataGenerator(camara_data_path, work_dir, seed).process()

    # Preprocess ALMG data
    AlmgDataPreprocessor(
        proposition_detail_path,
        proposition_text_path,
        legislative_sessions_path,
        work_dir
    ).process()

    # Generate ALMG data
    AlmgDataGenerator(work_dir, seed).process()

    # Treino - test split
    split_save_data(data_path, seed)

    # Executes model training via subprocesses
    # The operating system handles clearing the GPU after training ends
    scripts = ["train_mlm", "train_bi_encoder", "train_cross_encoder"]

    for s in scripts:
        args = [
                "python3",
                "-m",
                f"training.{s}",
                "--work_dir", work_dir,
                "--seed", str(seed),
                "--contex_size", str(contex_size)
            ]
        if s == "train_mlm":
            args.append("--base_model")
            args.append(base_model)

        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line.rstrip())
        returncode = process.wait()


        if returncode != 0:
            raise RuntimeError(
                "Subprocess failed (exit code {})".format(returncode)
            )
    # End of model training
    
    # Evaluate trained models
    # Compare with baselines
    os_operador = OpenSearchOperator(os_host, os_port, os_user, os_password)
    ModelTester(work_dir, open_ai_index, os_operador, contex_size).execute_experiments()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo."
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="jhu-clsp/mmBERT-base",
        help="Base BERT model"
    )

    parser.add_argument(
        "--contex_size",
        type=str,
        default=1500,
        help="Contex window size"
    )
    
    parser.add_argument(
        "--open_ai_index",
        type=str,
        default="proposicao_full_openai_large",
        help="Name of the OpenAI embedding index"
    )
    
    parser.add_argument(
        "--os_host",
        type=str,
        default="localhost",
        help="OpenSearch host"
    )
    
    parser.add_argument(
        "--os_port",
        type=int,
        default=9200,
        help="OpenSearch port"
    )
    
    parser.add_argument(
        "--os_user",
        type=str,
        default="admin",
        help="OpenSearch username"
    )
    
    parser.add_argument(
        "--os_password",
        type=str,
        default="ALMG_02%Similaridade",
        help="OpenSearch password"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to be used"
    )
    
    args = parser.parse_args()

    dataset_path = snapshot_download(
        repo_id="lucas-lage/SB2-Dataset",
        repo_type="dataset",
    )
    
    camara_data_path = os.path.join(
        dataset_path,
        "raw/chamber_of_deputies/dataset_proposicoes.csv"
    )
    proposition_detail_path = os.path.join(
        dataset_path,
        "raw/almg/detalheProposicao.json.gz"
    )
    proposition_text_path = os.path.join(
        dataset_path,
        "raw/almg/textoProposicao.json.gz"
    )
    legislative_sessions_path = os.path.join(
        dataset_path,
        "raw/almg/sessoes-legislativas.json"
    )
    base_model = args.base_model
    contex_size = args.contex_size
    open_ai_index = args.open_ai_index
    os_host = args.os_host
    os_port = args.os_port
    os_user = args.os_user
    os_password = args.os_password
    seed = args.seed

    run(
        camara_data_path,
        proposition_detail_path,
        proposition_text_path,
        legislative_sessions_path,
        base_model,
        contex_size,
        open_ai_index,
        os_host,
        os_port,
        os_user,
        os_password,
        seed
    )
