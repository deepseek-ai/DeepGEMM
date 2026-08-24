#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

#include "csrc/utils/env.hpp"

namespace {

void expect_valid(const std::string_view input, const int expected) {
    const auto actual = deep_gemm::detail::parse_env_int(input);
    if (not actual.has_value() or *actual != expected) {
        std::cerr << "Expected valid integer " << expected << " for '" << input << "'\n";
        std::exit(1);
    }
}

void expect_invalid(const std::string_view input) {
    if (deep_gemm::detail::parse_env_int(input).has_value()) {
        std::cerr << "Expected invalid integer for '" << input << "'\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    expect_valid("0", 0);
    expect_valid("1", 1);
    expect_valid("20", 20);
    expect_valid(" 7", 7);
    expect_valid("7 ", 7);
    expect_valid("7\n", 7);
    expect_valid("-1", -1);
    expect_valid("+5", 5);
    expect_valid("-2147483648", std::numeric_limits<int>::min());
    expect_valid("2147483647", std::numeric_limits<int>::max());

    expect_invalid("");
    expect_invalid("   ");
    expect_invalid("+");
    expect_invalid("true");
    expect_invalid("false");
    expect_invalid("c++17");
    expect_invalid("0x10");
    expect_invalid("16abc");
    expect_invalid("3000000000");
    expect_invalid("-3000000000");
    expect_invalid("99999999999999999999");
}
