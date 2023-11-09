export PYTHONPATH=$PYTHONPATH:/home/Claude
export FLASK_ENV=development
export FLASK_APP=e2e_pipeline/api_server.py
flask run --host=0.0.0.0 --port 5000
