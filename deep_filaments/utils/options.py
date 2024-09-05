"""
The class for managing the different options and parameters
"""


class TrainParameters:
    """
    Class for the parameters for training with default values

    Attributes
    ----------
    learning_rate: float
        The initial learning rate
    loss: str
        The name of the loss function
    epochs: int
        The number of epochs
    file_prefix: str
        The output filename for the weights
    project_dir: str
        The working directory for the project
    save_best_val: bool
        Save best model following the validation loss
    no_da_on_test: bool
        If True ensure that no data augmentation is applied on test set
    """

    def __init__(self):
        self.learning_rate = 0.00001
        self.loss = "binary cross-entropy"
        self.epochs = 100
        self.file_prefix = None
        self.project_dir = "."
        self.save_best_val = False
        self.no_da_on_test = False


class DatasetParameters:
    """
    Class for the parameters of the dataset management

    Attributes
    ----------
    percents: float
        The percentage of the data dedicated to training
    data_augmentation: int
        The data augmentation kind
    input_data_noise: float
        The variance of the noise apply on the input data
    output_data_noise: float
        The variance of the noise apply on the output (target) data
    nb_workers: int
        The number of workers to load to data (see pyTorch doc)
    prefetch: int
        The number of prefetch elements from the database
    normalize: bool
        Do inflight normalization
    use_newpatches: bool
        Use the patches from Siouar way of building the dataset
    rnd_seed: int
        The seed for the random method which split the data into three sets
    filename: str
        The name of the input file with the data
    batch_size: int
        The size of one batch
    """

    def __init__(self):
        self.percents = 0.8
        self.data_augmentation = 0
        self.input_data_noise = 0.0
        self.output_data_noise = 0.0
        self.nb_workers = 0
        self.prefetch = 2
        self.normalize = False
        self.use_newpatches = False
        self.rnd_seed = 20
        self.filename = None
        self.batch_size = 64
