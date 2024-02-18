import sys

from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.ollama import Ollama

from db_explorer.my_logger import logger
from db_explorer.train import train_from_db, train_from_files
from db_explorer.utils import Utils


class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


FROM_FILES = True
FROM_DB = False

model = "llama2"
model = "mistral"
vn = MyVanna(config=dict(model=model, path="./.chroma"))
utils = Utils()


if FROM_FILES:
    train_from_files(vn, utils)
if FROM_DB:
    train_from_db(vn)

# q = "How many countries in oceania speak english?"
q = "What are the top 3 official languages spoken in the Oceania continent?"
logger.info("Asking a question: {}", q)
response = vn.ask(q, print_results=False)
if response:
    logger.info("Reponse: {}", response[0])
else:
    logger.warning("No reponse was given ☹️ {}", response)
