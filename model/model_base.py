import logging
import os
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)


class ModelBase:
    def __init__(self, config):
        self.config = config
        self.batch_size = config["BATCH_SIZE"]
        self.data_csv = config["DATA_CSV"]
        self.learning_rate = config["LEARNING_RATE"]
        self.dropout = config["DROPOUT"]
        self.channel_list = config["CHANNEL_LIST"]
        self.num_classes = config["NUM_CLASSES"]
        self.activation = config["ACTIVATION"]
        self.epochs = config["EPOCHS"]
        self.steps_per_epoch = config["STEPS_PER_EPOCH"]
        resize_config = config["TRANSFORM"]["RESIZE"]
        self.resize = (resize_config[0], resize_config[1])
        logging.info(config)
        if config["RUN_ENV"] == "aws":
            self.save_dir = config["AWS_PARA"]["DIR_OUT"]
        elif config["RUN_ENV"] == "local":
            self.save_dir = self.__create_run_dir(
                config["LOCAL_PARA"]["DIR_OUT"])
        else:
            run_env = config["RUN_ENV"]
            raise Exception(f"Incorrect RUN_ENV: {run_env}")

    def __model_compile(self):
        raise NotImplementedError

    def __callback_compile(self):
        raise NotImplementedError

    def __set_model_parameters(self, **kwargs):
        raise NotImplementedError

    def model_fit(self, data_train, data_valid, **kwargs):
        raise NotImplementedError

    def __create_run_dir(self, save_dir):
        """Creates a numbered directory named "run1". If directory "run1" already
        exists then creates directory "run2", and so on.

        Parameters
        ----------
        save_dir : str
            The root directory where to create the "run{number}" folder.

        Returns
        -------
        str
            The full path of the newly created "run{number}" folder.
        """
        tf.gfile.MakeDirs(save_dir)
        list_of_files = tf.gfile.ListDirectory(save_dir)
        i = 1
        while f"run{i}" in list_of_files:
            i += 1
        run_dir = os.path.join(save_dir, f"run{i}")
        tf.gfile.MakeDirs(run_dir)
        print("#" * 40)
        print(f"Saving summaries on {run_dir}")
        print("#" * 40)
        return run_dir
