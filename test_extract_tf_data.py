from tensorboard.data.experimental import ExperimentFromDev


def extract_tf_data():
    experiment_id = "c1KCv3X3QvGwaXfgX1c4tg"
    experiment = ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    print(df["run"].unique())
    print(df["tag"].unique())

    dfw = experiment.get_scalars(pivot=True)
    dfw
    pass


if __name__ == "__main__":
    extract_tf_data()
