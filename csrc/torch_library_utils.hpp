#pragma once

#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "utils/exception.hpp"

namespace deep_gemm::torch_utils {

inline std::optional<std::tuple<int, int, int>> list_to_recipe3(
    const c10::optional<std::vector<int64_t>>& recipe) {
    if (not recipe.has_value() or recipe->empty()) {
        return std::nullopt;
    }
    DG_HOST_ASSERT(recipe->size() == 3);
    return std::make_tuple(static_cast<int>((*recipe)[0]),
                           static_cast<int>((*recipe)[1]),
                           static_cast<int>((*recipe)[2]));
}

inline std::optional<std::tuple<int, int>> list_to_recipe2(
    const c10::optional<std::vector<int64_t>>& recipe) {
    if (not recipe.has_value() or recipe->empty()) {
        return std::nullopt;
    }
    DG_HOST_ASSERT(recipe->size() == 2);
    return std::make_tuple(static_cast<int>((*recipe)[0]), static_cast<int>((*recipe)[1]));
}

inline std::variant<std::tuple<int, int, int>, std::tuple<int, int>> list_to_recipe_variant(
    const std::vector<int64_t>& recipe) {
    DG_HOST_ASSERT(recipe.size() == 2 or recipe.size() == 3);
    if (recipe.size() == 2) {
        return std::make_tuple(static_cast<int>(recipe[0]), static_cast<int>(recipe[1]));
    }
    return std::make_tuple(static_cast<int>(recipe[0]),
                           static_cast<int>(recipe[1]),
                           static_cast<int>(recipe[2]));
}

inline std::tuple<int, int, int> list_to_tuple3(const std::vector<int64_t>& values) {
    DG_HOST_ASSERT(values.size() == 3);
    return std::make_tuple(static_cast<int>(values[0]),
                           static_cast<int>(values[1]),
                           static_cast<int>(values[2]));
}

inline std::optional<std::vector<int>> list_to_optional_vector_int(
    const c10::optional<std::vector<int64_t>>& values) {
    if (not values.has_value()) {
        return std::nullopt;
    }
    std::vector<int> out;
    out.reserve(values->size());
    for (const auto value : *values) {
        out.push_back(static_cast<int>(value));
    }
    return out;
}

} // namespace deep_gemm::torch_utils
