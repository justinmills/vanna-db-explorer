# Testing vanna.ai against a pg database

This uses ollama local models to do so.

To get this working, you'll need to download and set up
[ollama](https://ollama.ai/). Once the service is up and running, you should be
able to run `poetry install` to set up all the dependencies and `poetry shell`
to activate the venv.

## Shortcuts

    # Spin up the db to test against
    docker compose up -d

    # Get a psql prompt against the db to poke around manually
    docker compose exec -it db psql -d world

    # Reset the vector db and run the main test
    poe clean-run
