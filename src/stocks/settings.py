LOGGING = {
    'version': 1,
    'loggers': {},
    'formatters': {
        'visual': {
            'format': '[%(asctime)s %(levelname)s %(name)s] %(message)s'
        }
    },
    'handlers': {
        'console': {
            'formatter': 'visual',
            'class': 'logging.StreamHandler',
            'level': 'INFO'
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO'
    }
}

MODEL_LOCATION = 'model'
TRANSFORMER_LOCATION = 'transformer'
STOCKS_CSV = 'data/joinedChart.csv'
SPLITS_CSV = 'data/splits.csv'
MODEL_DAYS_PERIOD = 2
TRAIN_ITERATIONS = 5
