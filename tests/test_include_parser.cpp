#include <cassert>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "../csrc/utils/hash.hpp"
#include "../csrc/utils/lazy_init.hpp"
#include "../csrc/jit/include_parser.hpp"

namespace fs = std::filesystem;

struct TempTree {
    fs::path root = fs::temp_directory_path() / "deepgemm-include-parser-test";

    TempTree() {
        fs::remove_all(root);
        fs::create_directories(root);
    }

    ~TempTree() {
        fs::remove_all(root);
    }
};

static void write_file(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << content;
}

static fs::path make_tree(const fs::path& root, const std::string& linked_content) {
    const auto library_root = root / "library";
    const auto linked_root = root / "source" / "cute";
    write_file(library_root / "include" / "deep_gemm" / "main.cuh", "// main\n");
    write_file(linked_root / "detail" / "config.hpp", linked_content);
    fs::create_directory_symlink(linked_root, library_root / "include" / "cute");
    return library_root;
}

static std::string hash_tree(const fs::path& library_root) {
    deep_gemm::IncludeParser::prepare_init(library_root.string());
    deep_gemm::IncludeParser parser;
    return parser.get_hash_value("#include <deep_gemm/main.cuh>");
}

int main() {
    TempTree temp;
    const auto first = make_tree(temp.root / "first", "same\n");
    const auto relocated = make_tree(temp.root / "relocated", "same\n");

    const auto first_hash = hash_tree(first);
    assert(first_hash == hash_tree(relocated));

    write_file(temp.root / "relocated" / "source" / "cute" / "detail" / "config.hpp", "changed\n");
    assert(first_hash != hash_tree(relocated));
}
