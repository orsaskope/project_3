#include <iostream>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cerrno>

#include "ivfflat.hpp"

using namespace std;

void swapEndian(u_int32_t* value) {
    u_int32_t byte4 = (*value >> 24) & 0x000000FF;
    u_int32_t byte3 = (*value >> 8) & 0x0000FF00;
    u_int32_t byte2 = (*value << 8) & 0x00FF0000;
    u_int32_t byte1 = (*value << 24) & 0xFF000000;
    *value =  byte1 | byte2| byte3 | byte4;
}


MNISTData readInputMnist(FILE* fd) {
    u_int32_t magic_num, images, rows, columns;
    fread(&magic_num, sizeof(u_int32_t), 1, fd);
    fread(&images, sizeof(u_int32_t), 1, fd);
    fread(&rows, sizeof(u_int32_t), 1, fd);
    fread(&columns, sizeof(u_int32_t), 1, fd);

    swapEndian(&magic_num);
    swapEndian(&images);
    swapEndian(&rows);
    swapEndian(&columns);

    cout << "magic number: " << magic_num << "\nnumber of images: " << images
         << "\nnumber of rows: " << rows << "\nnumber of columns: " << columns << endl;

    int size = rows * columns;

    MNISTData data;
    data.magic_number = magic_num;
    data.number_of_images = images;
    data.n_rows = rows;
    data.n_cols = columns;
    data.image_size = size;
    data.images.resize(images, std::vector<unsigned char>(size));

    int i_images = static_cast<int>(images);
    for(int i = 0; i < i_images; i++) {
        int res = fread(data.images[i].data(), sizeof(unsigned char), size, fd);
        if (res != size) {
            cout << "error in reading mnist images of size " << size << endl;
            exit(errno);
        }
    }

    return data;
}



std::vector<std::vector<float>> readInputSift(FILE* fd) {
    // Get file size
    fseek(fd, 0, SEEK_END);
    long file_size = ftell(fd);
    fseek(fd, 0, SEEK_SET);

    // Each SIFT vector entry: 4 bytes (ID or dim) + 320 floats = 4 + 320*4 = 1284 bytes
    const long vector_size = 1284;
    const int D = 320;

    // Number of vectors in file
    long nv = file_size / vector_size;

    // Allocate dataset correctly: nv rows × 320 floats
    std::vector<std::vector<float>> dataset(nv, std::vector<float>(D));

    int32_t dim;

    for (long i = 0; i < nv; i++) {

        // Read the 4-byte dimension/id field (not used)
        if (fread(&dim, sizeof(dim), 1, fd) != 1) {
            perror("Error reading dimension/id field");
            break;
        }

        // Read the 320 SIFT floats
        if (fread(dataset[i].data(), sizeof(float), D, fd) != D) {
            perror("Error reading SIFT vector");
            break;
        }
    }

    return dataset;
}


SIFTData readInputSift2(FILE* fd) {
    SIFTData data;

    //Check the size of the file
    fseek(fd, 0, SEEK_END);  
    long file_size = ftell(fd);  
    fseek(fd, 0, SEEK_SET);  

    //Calculate the number of vectors based on file size
    long vector_size = 1284;  
    data.number_of_vectors = (file_size / vector_size);  

    if (file_size % vector_size != 0) {
        std::cerr << "File size is not a multiple of the expected vector size!" << std::endl;
        exit(1);
    }

    // Step 3: Resize the vectors container
    data.vectors.resize(data.number_of_vectors, std::vector<float>(320));

    // Step 4: Read the vectors
    int32_t dim;
    for (int i = 0; i < data.number_of_vectors; ++i) {
        // Read the dimension (it should be the same for all vectors in the SIFT dataset)
        int res = fread(&dim, sizeof(dim), 1, fd);
        if (res != 1) {
            std::cerr << "Error reading dimension for vector " << i << std::endl;
            exit(1);
        }

        if (i == 0) {
            // Check if the first vector's dimension is 320 (SIFT standard)
            if (dim != 320) {
                std::cerr << "Error: The first vector's dimension is not 320, it's " << data.v_dim << std::endl;
                exit(1);  // Exit if the dimension is not correct
            }
        }

        res = fread(data.vectors[i].data(), sizeof(float), 320, fd);
        if (res != data.v_dim) {
            std::cerr << "Error reading vector " << i << std::endl;
            exit(1);
        }
    }

    return data;
}

int main(int argc, char* argv[]) {
    Params* p = ArgsParser(argc, argv);
    initializeParams(p);
    printParameters(p);

    IVFFLAT* ivfflat = nullptr;

    FILE* fd = fopen(p->input.c_str(), "r");
    if (!fd) {
        perror("Failed to open input file");
        exit(errno);
    }

    // FILE* fdi = fopen(p->query.c_str(), "r");
    // if (fdi == NULL) {
    //     perror("Failed to open query file");
    //     exit(errno);
    // }


    if (p->algorithm == 2) {
        if (p->type == "mnist") {
            MNISTData mnist = readInputMnist(fd);
            // MNISTData query = readInputMnist(fdi);

            ivfflat = new IVFFLAT(p->seed, p->kclusters, p->nprobe, p->n, p->r, mnist.image_size);

            vector<vector<float>> image_float(mnist.number_of_images, vector<float>(mnist.image_size));
            for (int i = 0; i < mnist.number_of_images; ++i) {
                for (int j = 0; j < mnist.image_size; ++j) {
                    image_float[i][j] = static_cast<float>(mnist.images[i][j]); // change to float (dividing by 255 so that the numbers are from 0 to 1);
                }
            }

            // vector<vector<float>> query_float(query.number_of_images, vector<float>(query.image_size));
            // for (int i = 0; i < query.number_of_images; ++i) {
            //     for (int j = 0; j < query.image_size; ++j) {
            //         query_float[i][j] = static_cast<float>(query.images[i][j]); 
            //     }
            // }

            if (!p->range) p->r = 0;
            IvfflatSearch_KNN(image_float, ivfflat, p->o);
            cout << "retuned to search" << endl;

        }
        else if (p->type == "sift") {
            vector<vector<float>> dataset = readInputSift(fd);
            // vector<vector<float>> query = readInputSift(fdi);

            ivfflat = new IVFFLAT(p->seed, p->kclusters, p->nprobe, p->n, p->r, 320);
            if (!p->range)  p->r = 0;
            IvfflatSearch_KNN(dataset, ivfflat, p->o);
            cout << "retuned to search" << endl;
        }
    }
    delete p;
    return 0;
}