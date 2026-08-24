#pragma once

#include <charconv>
#include <optional>
#include <string_view>
#include <system_error>

namespace deep_gemm::detail {

static constexpr bool is_env_whitespace(const char value) noexcept {
    return value == ' ' or value == '\t' or value == '\n' or
           value == '\r' or value == '\f' or value == '\v';
}

static std::optional<int> parse_env_int(std::string_view input) noexcept {
    auto begin = input.data();
    auto end = begin + input.size();

    while (begin != end and is_env_whitespace(*begin))
        ++begin;
    while (begin != end and is_env_whitespace(*(end - 1)))
        --end;

    if (begin == end)
        return std::nullopt;

    // Unlike scanf and strtol, from_chars does not accept a leading plus.
    // Preserve that conventional integer spelling explicitly.
    if (*begin == '+') {
        ++begin;
        if (begin == end or *begin < '0' or *begin > '9')
            return std::nullopt;
    }

    int value;
    const auto [parsed_end, error] = std::from_chars(begin, end, value, 10);
    if (error != std::errc() or parsed_end != end)
        return std::nullopt;
    return value;
}

} // namespace deep_gemm::detail
