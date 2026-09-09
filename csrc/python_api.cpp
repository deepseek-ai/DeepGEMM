#include <torch/all.h>
#include "utils/registration.h"

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega.hpp"
#include "apis/sm90_mega.hpp"
#include "apis/runtime.hpp"

REGISTER_EXTENSION(_C_extension)
