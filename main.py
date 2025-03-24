import torch

def main():
    print("Hello from deep-learning!")
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print("MPS available")

if __name__ == "__main__":
    main()
