export PYTHONPATH=$PYTHONPATH:/usr/src/app
export FLASK_ENV=development
export FLASK_APP=./api.py
export FLASK_DEBUG=1
flask run --host=0.0.0.0 --port 6000