import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from mace.cli.run_train import main

if __name__ == "__main__":
    main()