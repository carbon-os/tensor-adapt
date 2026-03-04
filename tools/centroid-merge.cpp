#include <tensor/centroid/centroid_merge.hpp>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string snapshot_dir;
    std::string out_path;
    int k = 16;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--out" && i+1 < argc)      out_path     = argv[++i];
        else if (a == "--clusters" && i+1 < argc) k       = std::stoi(argv[++i]);
        else if (snapshot_dir.empty())        snapshot_dir = a;
        else {
            std::cerr << "usage: centroid-merge <centroids-dir> --out <path> [--clusters N]\n";
            return 1;
        }
    }

    if (snapshot_dir.empty() || out_path.empty()) {
        std::cerr << "usage: centroid-merge <centroids-dir> --out <path> [--clusters N]\n";
        return 1;
    }

    try {
        tensor::centroid::CentroidMerge::run(snapshot_dir, out_path, k);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}