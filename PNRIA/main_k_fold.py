from PNRIA.torch_c.dataset import BaseDataset, KFoldsController

from PNRIA.configs.config import load_yaml


if __name__ == "__main__":

    config = load_yaml("/home/cloud-user/work/Toolbox/PNRIA/configs/config_kfolds.yaml")
    config_kfold = config["kfold"]
    config_train = config["trainset"]
    config_valid = config["validset"]
    config_test = config["testset"]

    k_fold_controler = KFoldsController.from_config(config_kfold)
    splits = k_fold_controler.generate_kfold_splits(
        k_fold_controler.k, k_fold_controler.k_train
    )

    area_groups, fold_assignments = k_fold_controler.create_folds_random_by_area(
        k_fold_controler.k
    )

    for split in splits:

        train_split, valid_split, test_split = split

        config_train["fold_assignments"] = fold_assignments
        config_train["fold_list"] = train_split

        config_valid["fold_assignments"] = fold_assignments
        config_valid["fold_list"] = valid_split

        config_test["fold_assignments"] = fold_assignments
        config_test["fold_list"] = test_split

        dataset_train = BaseDataset.from_config(config_train)
        dataset_valid = BaseDataset.from_config(config_valid)
        dataset_test = BaseDataset.from_config(config_test)

        # Start training here
        print("split :", split)
