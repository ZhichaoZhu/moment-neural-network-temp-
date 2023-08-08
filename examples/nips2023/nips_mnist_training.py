from mnn import models, utils

class ClassicMnnTrain(utils.training_tools.TrainProcessCollections):
    def make_model(self, args):
        model = models.MnnMlp(**args.MODEL)
        return model

class MeanOnlyTrain(utils.training_tools.TrainProcessCollections):
    def make_model(self, args):
        model = models.MnnMlpMeanOnly(**args.MODEL)
        return model

class ANNTrain(utils.training_tools.TrainProcessCollections):
    def make_model(self, args):
        model = models.AnnMlp(**args.MODEL)
        return model

def train_classic_mnn(args):
    args.save_name = 'classic_mnn'
    args.scale_factor = 0.1
    args.input_prepare = 'flatten_poisson'
    utils.training_tools.general_train_pipeline(args, train_func=ClassicMnnTrain)

def train_mnn_mean_only(args):
    args.save_name = 'mnn_mean_only'
    args.scale_factor = 1.
    args.input_prepare = None
    utils.training_tools.general_train_pipeline(args, train_func=MeanOnlyTrain)

def train_ann_mlp(args):
    args.save_name = 'ann_mlp'
    args.scale_factor = 1.
    args.input_prepare = None
    utils.training_tools.general_train_pipeline(args, train_func=ANNTrain)


if __name__ == '__main__':
    config = utils.training_tools.deploy_config()
    config.MODEL = {
        'structure': [28*28, 1000, 10],
        'num_class': 10,
        'predict_bias': False,
    }
    train_classic_mnn(config)
    train_mnn_mean_only(config)
    train_ann_mlp(config)