import sys
import numpy as np
import struct

# ============================= BUILD =============================

class BuildParams:
    def __init__(self):
        self.input = None
        self.index_path = None
        self.type = None

        # Initialize with default parameters 
        self.knn = 10
        self.m = 100
        self.imbalance = 0.03
        self.kahip_mode = 2
        self.layers = 3
        self.nodes = 64
        self.epochs = 10
        self.batch_size = 320
        self.lr = 0.001
        self.seed = 1
        self.dropout = 0.2      # ποσοστό dropout
        self.batchnorm = False  # αν θα χρησιμοποιηθεί batchnorm

def BuildParser(argv):
    p = BuildParams()

    i = 1
    while (i < len(argv)):
        arg = argv[i]

        if arg == "-d":
            i += 1
            validArgument(argv, i)
            p.input = argv[i]
        elif arg == "-i":
            i += 1
            validArgument(argv, i)
            p.index_path = argv[i]
        elif arg == "-type":
            i+= 1
            validArgument(argv, i)
            if argv[i] not in ("mnist", "sift"):
                print("Invalid argument for parameter -type")
                sys.exit(1)
            p.type = argv[i]
        elif arg == "--knn":
            i+=1
            validArgument(argv, i)
            p.knn = int(argv[i])
        elif arg == "-m":
            i+=1
            validArgument(argv, i)
            p.m = int(argv[i])
        elif arg == "--imbalance":
            i+=1
            validArgument(argv, i)
            p.imbalance = float(argv[i])
        elif arg == "--kahip_mode":
            i+=1
            if argv[i] == "0":  p.kahip_mode = 0
            elif argv[i] == "1":  p.kahip_mode = 1
            elif argv[i] == "2":  p.kahip_mode = 2
            else:
                print("Invalid kahip_mode argument")
                sys.exit(1)
        elif arg == "--layers":
            i+=1
            validArgument(argv, i)
            p.layers = int(argv[i])
        elif arg == "--nodes":
            i+=1
            validArgument(argv, i)
            p.nodes = int(argv[i])
        elif arg == "--epochs":
            i+=1
            validArgument(argv, i)
            p.epochs = int(argv[i])
        elif arg == "--batch_size":
            i+=1
            validArgument(argv, i)
            p.batch_size = int(argv[i])
        elif arg == "--lr":
            i+=1
            validArgument(argv, i)
            p.lr = float(argv[i])
        elif arg == "--seed":
            i+=1
            validArgument(argv, i)
            p.seed = int(argv[i])
        i += 1
    
    if p.input is None:
        print("there is no input file")
        sys.exit(1)
    if p.index_path is None:
        print("there is no index path")
        sys.exit(1)
    if p.type is None:
        print("there is not a given type")
        sys.exit(1)
    return p

# ============================= SEARCH =============================

class SearchParams:
    def __init__(self):
        self.input = None
        self.query = None
        self.path = None
        self.output = None
        self.type = None

        # Initialize with default parameters
        self.N = 1
        self.R = 0 # Will be initialized in parser, after reading the type
        self.T = 5
        self.range = True

def SearchParser(argv):
    p = SearchParams()

    i = 1
    while (i < len(argv)):
        arg = argv[i]

        if arg == "-d":
            i += 1
            validArgument(argv, i)
            p.input = argv[i]
        elif arg == "-q":
            i += 1
            validArgument(argv, i)
            p.query = argv[i]
        elif arg == "-i":
            i += 1
            validArgument(argv, i)
            p.path = argv[i]
        elif arg == "-o":
            i += 1
            validArgument(argv, i)
            p.output = argv[i]
        elif arg == "-type":
            i+= 1
            validArgument(argv, i)
            if argv[i] not in ("mnist", "sift"):
                print("Invalid argument for parameter -type")
                sys.exit(1)
            p.type = argv[i]
        elif arg == "-N":
            i += 1
            validArgument(argv, i)
            p.N = int(argv[i])
        elif arg == "-R":
            i += 1
            validArgument(argv, i)
            p.R = int(argv[i])
        elif arg == "-T":
            i += 1
            validArgument(argv, i)
            p.T = int(argv[i])
        elif arg == "-range":
            i += 1
            validArgument(argv, i)
            if argv[i] in ("True", "true"):    p.range = True
            elif argv[i] in ("False","false"): p.range = False
            else:
                print("Invalid argument for parameter -range")
                sys.exit(1)
        i += 1
    
    if p.input is None: 
        print("there is no input file")
        sys.exit(1)
    if p.query is None: 
        print("there is no query file")
        sys.exit(1)
    if p.path is None: 
        print("there is no path")
        sys.exit(1)
    if p.output is None:
        print("there is no output file")
        sys.exit(1)
    if p.type is None: 
        print("there is not a given type")
        sys.exit(1)

    if p.R == 0:
        if p.type == "sift":
            p.R = 2800
        else:
            p.R = 2000
    return p

def validArgument(argv, i):
    if i >= len(argv):
        print("Not enough values given")
        sys.exit(1)

    if argv[i].startswith("-"):
        print(f"Invalid argument: {argv[i]}")
        sys.exit(1)

# ============================= MNIST PARSER =============================
def swap_endian32(x: int) -> int:
    byte1 = (x >> 24) & 0x000000FF
    byte2 = (x >> 8)  & 0x0000FF00
    byte3 = (x << 8)  & 0x00FF0000
    byte4 = (x << 24) & 0xFF000000
    return byte1 | byte2 | byte3 | byte4

class MnistData:
    def __init__(self, magic_number, number_of_images, n_rows, n_cols, images):
        self.magic_number = magic_number
        self.number_of_images = number_of_images
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.image_sz = n_rows * n_cols
        self.images = images    # images: array shape (number_of_images, image_s), dtype=uint8ize)

def read_mnist(path):
    fd = open(path, "rb")

    magic_number = int.from_bytes(fd.read(4), "big")
    number_of_images = int.from_bytes(fd.read(4), "big")
    n_rows = int.from_bytes(fd.read(4), "big")
    n_cols = int.from_bytes(fd.read(4), "big")

    image_sz = n_rows * n_cols
    dataset = []

    for i in range(number_of_images):
        img = fd.read(image_sz)
        if len(img) != image_sz:
            print("Error reading MNIST image")
            exit(1)
        dataset.append(list(img))

    fd.close()
    return MnistData(magic_number, number_of_images, n_rows, n_cols, dataset)

# ============================= SIFT PARSER =============================

class SiftData:
    def __init__(self, number_of_vectors, dimension, dataset):
        self.count = number_of_vectors
        self.dim = dimension
        self.dataset = dataset  # list[list[float]]

def read_sift(path):
    fd = open(path, "rb")

    dataset = []

    # Find dataset size
    fd.seek(0, 2)
    file_size = fd.tell()
    fd.seek(0)

    entry_size = 4 + 320 * 4  # 4 bytes for dim, 320 floats
    count = file_size // entry_size

    for i in range(count):
        dim_bytes = fd.read(4)
        if len(dim_bytes) < 4:
            print("Error reading SIFT header")
            exit(1)

        dim = struct.unpack("<i", dim_bytes)[0]   # LITTLE ENDIAN
        if dim != 320:
            print("Invalid SIFT dimension, expected 320")
            exit(1)

        vec_bytes = fd.read(320 * 4)
        if len(vec_bytes) < 320 * 4:
            print("Error reading SIFT vector")
            exit(1)

        vec = list(struct.unpack("<320f", vec_bytes))  # LITTLE ENDIAN FLOATS
        dataset.append(vec)

    fd.close()
    return SiftData(count, 320, dataset)
