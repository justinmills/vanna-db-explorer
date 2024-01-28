from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

from db_explorer.my_logger import logger
from db_explorer.ollama_vanna import OllamaVannaLLM
from db_explorer.train import train_from_files
from db_explorer.utils import Utils


class MyVanna(ChromaDB_VectorStore, OllamaVannaLLM):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OllamaVannaLLM.__init__(self, config=config)


vn = MyVanna(config=dict(model="llama2", path="./.chroma"))
utils = Utils()

train_from_files(vn, utils)

# q = "How many countries in oceania speak english?"
q = "What are the top 3 official languages spoken in Oceania"
logger.info("Asking a question: {}", q)
response = vn.ask(q, print_results=False)
if response:
    logger.info("Reponse: {}", response[0])
else:
    logger.warning("No reponse was given ☹️ {}", response)
