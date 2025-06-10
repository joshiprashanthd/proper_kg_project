import sys
import time
sys.path.append("..")

from src.components.triplets_extractor import TripletsExtractor

def timeit(func):
    start = time.time()
    func()
    end = time.time()
    return end - start

triplets_extractor = None

def load_extractor():
    global triplets_extractor
    triplets_extractor = TripletsExtractor()


def extracting_triplets():
    global triplets_extractor
    results = triplets_extractor.extract_triplets("depression", top_k=40)

if __name__ == "__main__":
    load_time = timeit(load_extractor)
    print(f"Time taken for loading extractor: {load_time}")

    avg_time = 0
    for i in range(10):
        avg_time += timeit(extracting_triplets)
    avg_time /= 10
    print(f"Average time taken for extracting top 40 triplets: {avg_time}")
    

# OUTPUT:
# Time taken for loading extractor: 238.6182074546814 ~ 240 sec ~ 4 min
# Average time taken for extracting top 40 triplets: 8.338587212562562  ~ 8 sec