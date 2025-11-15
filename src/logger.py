
import wandb as wnb


class WandBLogger:
    """
    Interface for weights and biases logger.
    """

    def __init__(
        self,
        group,
        name,
        local_dir=".",
        job_type=None,
        entity="bnns-for-molecules",
        project_name="semi-supervised-ensembles",
        disable=False,
    ):
        """
        Arguments:
            hydra_config: Hydra config object as dict.
            group: Specifies group under which to store the run (Could for example be objective problem name).
            name: Given name to the run. Should be relatively short for visability.
            job_type: Can be used to filter runs, for example "train" or "eval".
            entity: Name of the entity to which the project and run belongs to. Should be name of our team on W&B (default is correct).
            project_name: Name of the project which the run is logged under. I suggest we keep default.
            dir: Local directory in which to save local wandb files.
            disable: disable logging completely (useful for test runs).
        """
        self.disable = disable

        self.name = name
        self.group = group
        self.project_name = project_name
        # Where to save local wnb files.
        self.dir = local_dir
        self.entity = entity
        self.job_type = job_type

    def init_run(self, hparams):
        # self.name += "_seed=" + str(hparams['seed'])
        if not self.disable:
            self.run = wnb.init(
                name=self.name,
                group=self.group,
                config=hparams,
                project=self.project_name,
                dir=self.dir,
                #entity=self.entity,
                job_type=self.job_type,
            )

    @property
    def run_id(self):
        if not self.disable:
            return self.run.id
        else:
            return None

    def log(self, key, value, step=None):
        """
        Logs a scalar with name key.
        Can also use step to log during training.
        """
        if not self.disable:
            self.run.log({key: value})

    def log_dict(self, result_dict, step=None):
        """
        Logs a scalar with name key.
        Can also use step to log during training.
        """
        if not self.disable:
            self.run.log(result_dict)

    def watch(self, model, log="all"):
        if not self.disable:
            wnb.watch(model, log=log)

    def end_run(self):
        if not self.disable:
            self.run.finish()
