from argparse import ArgumentParser
from logging.config import dictConfig

from stocks.settings import LOGGING


def main():
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest='action')
    sub.add_parser('train', help='Train model. Should be executed before predictions')
    server = sub.add_parser('server', help='Run web server to allow predictions')
    server.add_argument('-p', '--port', type=int, default=8000)
    args = parser.parse_args()
    dictConfig(LOGGING)
    if args.action == 'train':
        from stocks.modeling.create_model import ModelCreator
        ModelCreator().train_model()
    else:
        from stocks.server.app import app
        try:
            app.run(host='0.0.0.0', port=args.port)
        except Exception:
            pass


if __name__ == '__main__':
    main()
