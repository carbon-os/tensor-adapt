#include <tensor/centroid/centroid_merge.hpp>
#include <tensor/centroid/centroid_accumulator.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace tensor::centroid {

void CentroidMerge::run(
    const std::string& snapshot_dir,
    const std::string& out_path,
    int k)
{
    // Read first file to determine D.
    int D = 0;
    for (const auto& e : fs::directory_iterator(snapshot_dir)) {
        if (e.path().extension() == ".centroid") {
            std::ifstream f(e.path(), std::ios::binary);
            f.read(reinterpret_cast<char*>(&D), sizeof(int));
            break;
        }
    }
    if (D == 0) throw std::runtime_error("No .centroid files in " + snapshot_dir);

    // Reuse accumulator merge logic.
    CentroidAccumulator acc(D, snapshot_dir);
    acc.merge_and_write(k, out_path);
}

} // namespace tensor::centroid