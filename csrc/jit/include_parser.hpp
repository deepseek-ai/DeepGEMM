#pragma once

#include <filesystem>
#include <regex>
#include <string>
#include <vector>

#include "../utils/format.hpp"
#include "../utils/system.hpp"

namespace deep_gemm {

class IncludeParser {
    std::unordered_map<std::string, std::optional<std::string>> cache;

    static std::vector<std::string> get_includes(const std::string& code, const std::filesystem::path& file_path = "") {
        std::vector<std::string> includes;
        const std::regex pattern(R"(#\s*include\s*[<"][^>"]+[>"])");
        std::sregex_iterator iter(code.begin(), code.end(), pattern);
        const std::sregex_iterator end;

        // TODO: parse relative paths as well
        for (; iter != end; ++ iter) {
            const auto include_str = iter->str();
            const int len = include_str.length();
            if (include_str.substr(0, 10) == "#include <" and include_str[len - 1] == '>' and include_str[10] != ' ' and include_str[len - 2] != ' ') {
                std::string filename = include_str.substr(10, len - 11);
                if (filename.substr(0, 9) == "deep_gemm")  // We only parse `<deep_gemm/*>`
                    includes.push_back(filename);
            } else {
                std::string error_info = fmt::format("Non-standard include: {}", include_str);
                if (file_path != "")
                    error_info += fmt::format(" ({})", file_path.string());
                DG_HOST_UNREACHABLE(error_info);
            }
        }
        return includes;
    }

public:
    static std::filesystem::path library_include_path;

    static void prepare_init(const std::string& library_root_path) {
        library_include_path = std::filesystem::path(library_root_path) / "include";
    }

    std::string get_hash_value(const std::string& code, const bool& exclude_code = true) {
        std::stringstream ss;
        for (const auto& i: get_includes(code))
            ss << get_hash_value_by_path(library_include_path / i) << "$";
        if (not exclude_code)
            ss << "#" << get_hex_digest(code);
        return get_hex_digest(ss.str());
    }

    std::string get_hash_value_by_path(const std::filesystem::path& path) {
        // Check whether hit in cache
        // ReSharper disable once CppUseAssociativeContains
        if (cache.count(path) > 0) {
            const auto opt = cache[path];
            if (not opt.has_value())
                DG_HOST_UNREACHABLE(fmt::format("Circular include may occur: {}", path.string()));
            return opt.value();
        }

        // Read file and calculate hash recursively
        std::ifstream in(path);
        if (not in.is_open())
            DG_HOST_UNREACHABLE(fmt::format("Failed to open: {}", path.string()));
        std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        cache[path] = std::nullopt;
        return (cache[path] = get_hash_value(code, false)).value();
    }
};

DG_DECLARE_STATIC_VAR_IN_CLASS(IncludeParser, library_include_path);

static auto include_parser = std::make_shared<IncludeParser>();

}  // namespace deep_gemm
