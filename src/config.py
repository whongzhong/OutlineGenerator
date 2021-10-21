import argparse


def ComGen_config():
    parser = argparse.ArgumentParser()
    # OS config
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_save", type=str)
    # Prepare data
    parser.add_argument("--build_data", action="store_true")
    parser.add_argument("--train_save", type=str)
    parser.add_argument("--valid_save", type=str)
    parser.add_argument("--test_save", type=str)
    # Training config
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--pretrain_cp", type=str, default="facebook/bart-base")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.000003)
    parser.add_argument("--fix_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    # Predict config
    parser.add_argument("--predict", action="store_true")
    # Debug config
    parser.add_argument("--mini_test", action="store_true")

    args = parser.parse_args()
    return args


def Ordering_config():
    parser = argparse.ArgumentParser()
    # OS config
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_save", type=str)
    # Prepare data
    parser.add_argument("--build_data", action="store_true")
    parser.add_argument("--train_save", type=str)
    parser.add_argument("--valid_save", type=str)
    parser.add_argument("--test_save", type=str)
    # Training config
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.000003)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--fix_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--pretrain_path", type=str)

    args = parser.parse_args()
    
    return args


def Base_config():
    parser = argparse.ArgumentParser()
    # OS config
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_save", type=str)
    # Prepare data
    parser.add_argument("--build_data", action="store_true")
    parser.add_argument("--train_save", type=str)
    parser.add_argument("--valid_save", type=str)
    parser.add_argument("--test_save", type=str)
    # Training config
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.000003)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--fix_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--pretrain_path", type=str)

    args = parser.parse_args()
    
    return args