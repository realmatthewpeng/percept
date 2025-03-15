import argparse
import logging

import classifier_evaluator as ceval
import phosphene_generator as pgen
import utils


def main(args):
    if args.evaluate is not None:
        classifier_path, dataset_X, dataset_Y, xdim, ydim = args.evaluate
        classifier = utils.get_pretrained_classifier(classifier_path)
        _, _, test_images, test_labels = utils.get_processed_dataset(test_only=True, test_X_path=dataset_X, test_Y_path=dataset_Y, xdim=int(xdim), ydim=int(ydim))
        ceval.eval_model(classifier, test_images, test_labels)
        return

    cfg = utils.load_config(args.config)

    model = utils.get_percept_model(cfg['phosphene_generator'])
    logging.debug(model)

    implant = utils.get_implant(cfg['phosphene_generator'])
    logging.debug(implant)

    image_preprocessor = None
    if (('image_preprocessor' in cfg) and (cfg['image_preprocessor'] is not None)):
        image_preprocessor = utils.get_image_preprocessor(cfg['image_preprocessor'])
    logging.debug(image_preprocessor)
    
    if (('pretrained_classifier' in cfg) and (cfg['pretrained_classifier'] is not None)):
        _, _, pre_test_images, pre_test_labels = utils.get_dataset(cfg['phosphene_generator'], test_only=True)

        xdim, ydim = pgen.generate_percept([], [], pre_test_images, pre_test_labels, implant, model, image_preprocessor)

        _, _, post_test_images, post_test_labels = utils.get_processed_dataset(test_only=True, xdim=xdim, ydim=ydim)

        trained_classifier = utils.get_pretrained_classifier(cfg['pretrained_classifier'])

        ceval.eval_model(trained_classifier, post_test_images, post_test_labels)

    else:
        pre_train_images, pre_train_labels, pre_test_images, pre_test_labels = utils.get_dataset(cfg['phosphene_generator'])

        xdim, ydim = pgen.generate_percept(pre_train_images, pre_train_labels, pre_test_images, pre_test_labels, implant, model, image_preprocessor)

        classifier = utils.get_classifier(cfg['classifier_evaluator'])
        logging.debug(classifier)

        post_train_images, post_train_labels, post_test_images, post_test_labels = utils.get_processed_dataset(xdim=xdim, ydim=ydim)

        ceval.train_model(classifier, post_train_images, post_train_labels)

        trained_classifier = utils.get_trained_classifier(cfg['classifier_evaluator'])

        ceval.eval_model(trained_classifier, post_test_images, post_test_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--config", type=str, default=None,
                       help="config file (yaml) with the pipeline configurations: e.g. '_config.yaml' ")
    group.add_argument("-e", "--evaluate", nargs=5, metavar=('classifier', 'dataX', 'dataY', 'xdim', 'ydim'), default=None,
                       help="evaluation only mode for a pretrained classifier on a processed dataset. Provide the path to the classifier, dataX, and dataY files."
                       " Also provide the dimensions of the processed dataset: xdim and ydim.")

    args = parser.parse_args()
    main(args)