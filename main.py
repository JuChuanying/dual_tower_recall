import argparse
from src.train import main as train_main
from src.inference import demo_inference

def main():
    parser = argparse.ArgumentParser(description='双塔召回推荐系统')
    parser.add_argument('--mode', choices=['train', 'inference'], 
                       default='train', help='运行模式')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_main()
    elif args.mode == 'inference':
        demo_inference()

if __name__ == "__main__":
    main()
