from vanna.base import VannaBase

from db_explorer.my_logger import logger
from db_explorer.utils import Utils


def train_from_db(vn: VannaBase) -> None:
    pg_host = "localhost"
    pg_db = "world"
    pg_user = "postgres"
    pg_pass = "postgres"
    pg_port = 8000

    # NOTE: Keeping this connection seems to break things when generating plotly
    # code...I think it generates a bad query and that breaks it.
    # logger.info("Connecting to postgres")
    vn.connect_to_postgres(
        host=pg_host, dbname=pg_db, user=pg_user, password=pg_pass, port=pg_port
    )

    # The information schema query may need some tweaking depending on your database. This is a good starting point.
    # logger.info("Running sql to list public schema columns")
    # df_information_schema = vn.run_sql(
    #     "SELECT * FROM information_schema.columns WHERE table_schema = 'public'"
    # )
    # # Now generate a plan from the above schema. This will break up the information
    # # schema into bite-sized chunks that can be referenced by the LLM
    # logger.info("Generating a training plan from the info schema")
    # plan = vn.get_training_plan_generic(df_information_schema)
    # logger.info("Got a training plan with {} lines", len(plan.get_summary()))
    # LOG_PLAN = False
    # if LOG_PLAN:
    #     for line in plan.get_summary():
    #         logger.debug("Training plan summary line: {}", line)
    #         pass
    # # If you like the plan, then uncomment this and run it to train
    # logger.info("Training with that plan")
    # vn.train(plan=plan)

    # logger.info("Training with a sql statement")
    # vn.train(
    #     sql="select * from country c join countrylanguage l on c.code = l.countrycode;"
    # )


def train_from_files(vn: VannaBase, utils: Utils) -> None:
    # I swiped this from this repo:
    # https://github.com/r0mymendez/text-to-sql/blob/main/src/main.py Super handy
    # stuff and it wound up working quite a bit better. Still struggling with the
    # plotly stuff though. But if I disconnect the database above it works just
    # fine.
    logger.info("Train with ddl")
    vn.train(ddl=utils.read_files(path="training/ddl", file_type="sql"))
    logger.info("Train with docs")
    vn.train(documentation=utils.read_files(path="training/docs", file_type="md"))
    logger.info("Train with sql")
    vn.train(sql=utils.read_files(path="training/docs", file_type="sql"))
