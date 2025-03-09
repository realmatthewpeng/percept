import argparse
import logging

import classifier_evaluator as ceval
import phosphene_generator as pgen
import utils


def main(args):
    cfg = utils.load_config(args.config)

    model = utils.get_percept_model(cfg['phosphene_generator'])
    logging.debug(model)

    implant = utils.get_implant(cfg['phosphene_generator'])
    logging.debug(implant)
    
    if (('pretrained_classifier' in cfg) and (cfg['pretrained_classifier'] is not None)):
        _, _, pre_test_images, pre_test_labels = utils.get_dataset(cfg['phosphene_generator'], test_only=True)

        pgen.generate_percept([], [], pre_test_images, pre_test_labels, implant, model)

        _, _, post_test_images, post_test_labels = utils.get_processed_dataset(test_only=True)

        trained_classifier = utils.get_pretrained_classifier(cfg['pretrained_classifier'])

        ceval.eval_model(trained_classifier, post_test_images, post_test_labels)

    else:
        pre_train_images, pre_train_labels, pre_test_images, pre_test_labels = utils.get_dataset(cfg['phosphene_generator'])

        pgen.generate_percept(pre_train_images, pre_train_labels, pre_test_images, pre_test_labels, implant, model)

        classifier = utils.get_classifier(cfg['classifier_evaluator'])
        logging.debug(classifier)

        post_train_images, post_train_labels, post_test_images, post_test_labels = utils.get_processed_dataset()

        ceval.train_model(classifier, post_train_images, post_train_labels)

        trained_classifier = utils.get_trained_classifier(cfg['classifier_evaluator'])

        ceval.eval_model(trained_classifier, post_test_images, post_test_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c", "--config", type=str, default=None,
                       help="config file (yaml) with the pipeline configurations: e.g. '_config.yaml' ")

    args = parser.parse_args()
    main(args)