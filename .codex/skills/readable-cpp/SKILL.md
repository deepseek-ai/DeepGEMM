---
name: readable-cpp
description: Readable C/C++/Rust/CUDA code rules inspired by The Art of Readable Code. Use when writing, reviewing, or refactoring C, C++, Rust, or CUDA code. Enforces short functions, flat control flow, clear naming, readable structure, and idiomatic patterns.
---

# Readable C/C++/Rust/CUDA Rules (/readable-cpp)

Apply these rules when writing, reviewing, or refactoring C, C++, Rust, or CUDA code. Inspired by *The Art of Readable Code* by Dustin Boswell and Trevor Foucher.

**Core principle: Code should be easy to understand.** The time it takes someone else (or future you) to understand the code is the ultimate metric.

## 1. Keep Functions Short and Focused

- A function should do **one thing**. If you can describe what it does with "and", split it.
- Aim for functions that fit on one screen (~15-25 lines). If it's longer, extract sub-tasks.
- Each function should operate at a **single level of abstraction** — don't mix high-level logic with low-level details in the same function.

## 2. Flatten Control Flow — No Deep Nesting

- **Never nest more than 2 levels deep.** If you have a loop inside a loop, or an `if` inside a loop inside an `if`, extract the inner block into a helper function with a descriptive name.
- Use **early returns / guard clauses** to handle edge cases at the top, keeping the main logic flat.
- Prefer `continue` or `break` to skip iterations rather than wrapping the body in a conditional.
- Replace complex conditionals with well-named helper functions or variables that explain the intent.

```cpp
// Bad: nested and hard to follow
for (auto& user : users) {
    if (user.is_active()) {
        for (auto& order : user.orders()) {
            if (order.is_pending()) {
                process(order);
            }
        }
    }
}

// Good: flat, each function name explains what it does
auto active_users = get_active_users(users);
for (auto& user : active_users) {
    process_pending_orders(user.orders());
}
```

## 3. Name Things Clearly

- **Pack information into names.** Use specific, concrete words — `fetch_page` not `get`, `num_retries` not `n`.
- **Avoid generic names** like `tmp`, `data`, `result`, `val`, `info`, `handle` — unless the scope is tiny (2-3 lines).
- **Use names that can't be misconstrued.** If a range is inclusive, say `max_items` not `limit`. If a boolean, use `is_`, `has_`, `should_`, `can_` prefixes.
- **Match the name length to the scope.** Short names for small scopes, descriptive names for wide scopes.
- **Don't use abbreviations** unless they're universally understood (`num`, `max`, `min`, `err` are fine; `svc_mgr_cfg` is not).

## 4. Make Control Flow Easy to Follow

- Put the **changing/interesting value on the left** side of comparisons: `if (length > 10)` not `if (10 < length)`.
- Order `if/else` blocks: **positive case first**, simpler case first, or the more interesting case first.
- Minimize the number of variables the reader has to track. Reduce the **mental footprint** of each block.
- Avoid deeply nested ternary operators — if it's not immediately obvious, use an `if/else`.

## 5. Break Down Giant Expressions

- Use **explaining variables** to break complex expressions into named pieces.
- Use **summary variables** to capture a long expression that's used more than once.
- Apply **De Morgan's laws** to simplify negated boolean expressions.

```cpp
// Bad
if (!(age >= 18 && has_id && !is_banned)) {
    deny();
}

// Good
bool is_eligible = age >= 18 && has_id && !is_banned;
if (!is_eligible) {
    deny();
}
```

## 6. Extract Unrelated Subproblems

- If a block of code is solving a **subproblem unrelated to the main goal** of the function, extract it.
- The helper function should be **pure and self-contained** — it shouldn't need to know about the calling context.
- This is the single most effective way to improve readability: separate *what* you're doing from *how*.

## 7. One Task at a Time

- Each section of code should do **one task**. If a function is doing parsing AND validation AND transformation, split them into separate steps.
- List the tasks a function does. If there's more than one, reorganize so each task is in its own block or function.

## 8. Reduce Variable Scope

- **Declare variables close to where they're used.** Don't declare at the top of a function if it's only used 30 lines later.
- **Minimize the "live time" of a variable** — the fewer lines between its assignment and last use, the easier it is to follow.
- **Prefer write-once variables.** Variables that are assigned once and never modified are easier to reason about.
- **Eliminate unnecessary variables.** If a variable is used only once and doesn't clarify anything, inline it.

## 9. No Magic Numbers or Strings

- Replace **magic numbers and strings** with named constants: `if (retries > MAX_RETRIES)` not `if (retries > 3)`.
- If a value has meaning, give it a name. The name documents the intent.
- Group related constants together.

## 10. Fewer Function Arguments

- Aim for **3 or fewer arguments** per function. More than that is a smell.
- Group related arguments into a **struct, class, or tuple**.
- If a function needs many config-like options, pass a single config/options object.
- Boolean flag arguments are a sign the function does two things — split it instead.

## 11. Consistency

- If the codebase does something one way, **do it the same way**. Don't mix styles.
- Consistent naming patterns, consistent structure, consistent error handling.
- When joining an existing codebase, **match the existing conventions** even if you'd prefer a different style.
- Surprise is the enemy of readability — predictable code is readable code.

## 12. Write Less Code

- The best code is **no code at all**. Question whether a feature is truly needed before implementing.
- **Don't over-engineer.** Solve the problem at hand, not hypothetical future problems.
- Remove dead code. Commented-out code is dead code.
- Use standard libraries before writing custom solutions.

## 13. Comments: Explain Why, Not What

- Don't comment **what** the code does — the code already says that. Comment **why** it does it.
- Comment **flaws and workarounds**: `// TODO:`, `// HACK:`, `// XXX:` with explanation.
- Comment **surprising behavior** or non-obvious decisions — things where a reader would ask "why?".
- **Don't comment bad code — rewrite it.** If you need a comment to explain what a block does, extract it into a well-named function instead.

## 14. Design Code to Survive Auto-Formatting

- Write code that looks good **after** the auto-formatter runs. If a chained expression or repeated pattern would be broken across 4+ lines by the formatter, extract a helper function instead.
- **Prefer one-line helper calls** over long inline chains that the formatter will expand vertically.
- The formatter is your reader's first impression. Run it *before* committing — if the result looks ugly, that's a signal to refactor, not to disable the formatter.

```rust
// Bad: rustfmt expands this to 4 lines per field — noisy and repetitive
fn from_dict(cfg: &Bound<'_, PyDict>) -> PyResult<Self> {
    Ok(Self {
        rom: cfg.get_item("rom")?.ok_or_else(|| missing("rom"))?.extract()?,
        // ... each field becomes 4 lines after rustfmt
    })
}

// Good: extract a helper so each field stays one clean line
fn get_required<T: FromPyObject>(cfg: &Bound<'_, PyDict>, key: &str) -> PyResult<T> {
    cfg.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(key.to_string()))?
        .extract()
}

fn from_dict(cfg: &Bound<'_, PyDict>) -> PyResult<Self> {
    Ok(Self {
        rom: get_required(cfg, "rom")?,
        actions: get_required(cfg, "actions")?,
    })
}
```

```cpp
// Bad: clang-format wraps this into a hard-to-scan block
auto result = container.find(key)->second.get_value().transform(func).value_or(default_val);

// Good: name the intermediate step
auto& entry = container.find(key)->second;
auto result = entry.get_value().transform(func).value_or(default_val);
```

---

# C-Specific Rules

## 15. RAII-Like Patterns with goto Cleanup

- In C, use the **goto cleanup pattern** for resource management — allocate at the top, clean up at a single labeled block at the bottom.
- Never scatter `free()` calls across multiple return paths. A single cleanup section is easier to audit.
- Use `__attribute__((cleanup))` (GCC/Clang) when available for automatic cleanup.

```c
// Good: single cleanup path
int process_file(const char *path) {
    int ret = -1;
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    char *buf = malloc(BUF_SIZE);
    if (!buf) goto cleanup_file;

    // ... do work ...
    ret = 0;

cleanup_buf:
    free(buf);
cleanup_file:
    fclose(fp);
    return ret;
}
```

## 16. Use `const` Liberally

- Mark pointers `const` when the function doesn't modify the pointed-to data: `const char *msg`.
- Mark local variables `const` when they don't change after initialization.
- This documents intent and helps the compiler catch mistakes.

## 17. Prefer Sized Types for Data Structures

- Use `<stdint.h>` types (`uint32_t`, `int64_t`) for data that crosses boundaries (files, network, hardware).
- Use `size_t` for sizes and counts, `ptrdiff_t` for pointer differences.
- Use `int` and `unsigned` for simple loop counters and local arithmetic.

## 18. Defensive Macro Hygiene

- Wrap macro bodies in `do { ... } while(0)` for statement-like macros.
- Parenthesize all macro parameters: `#define SQUARE(x) ((x) * (x))`.
- Prefer `static inline` functions over macros when possible (type safety, debuggability).
- Use `_Generic` (C11) for type-safe "overloading" instead of macro tricks.

---

# C++-Specific Rules

## 19. Use RAII for All Resources

- Every resource (memory, file handles, locks, sockets) should be owned by an RAII object.
- Use `std::unique_ptr` for exclusive ownership, `std::shared_ptr` only when shared ownership is genuinely needed.
- Write custom RAII wrappers for non-standard resources (e.g., C library handles).
- Never use raw `new`/`delete` in application code — let smart pointers and containers handle it.

## 20. Prefer Value Semantics and Move

- Pass small objects by value, large objects by `const&`.
- Return objects by value — rely on RVO/NRVO and move semantics.
- Implement move constructors/assignment for types that own resources.
- Use `std::move` only when you truly want to transfer ownership — don't `std::move` from things you'll use again.

## 21. Use Modern C++ Over C Idioms

- Use `std::array` over C arrays, `std::string` over `char*`, `std::vector` over `malloc`/`realloc`.
- Use `std::optional` over sentinel values, `std::variant` over type-unsafe unions.
- Use range-based `for` loops: `for (const auto& item : container)`.
- Use structured bindings (C++17): `auto [key, value] = *map.begin();`.
- Use `std::format` (C++20) or `fmt::format` over `sprintf` / string concatenation.

## 22. Templates: Keep It Simple

- Use concepts (C++20) to constrain templates — errors become readable.
- Prefer `if constexpr` over SFINAE when possible.
- Don't write template metaprogramming unless the benefit is clear and the team can maintain it.
- A non-template solution that's slightly less generic is often better than a template solution nobody understands.

## 23. Use `constexpr` and `const` Aggressively

- Mark functions `constexpr` when they can be evaluated at compile time.
- Use `constexpr` variables instead of `#define` for constants.
- Use `const` on member functions that don't modify state.
- `consteval` (C++20) for functions that *must* be compile-time evaluated.

## 24. Error Handling: Pick One Pattern

- Use exceptions for truly exceptional conditions, `std::expected` (C++23) or `std::optional` for expected failures.
- Don't mix error codes and exceptions in the same layer.
- If using exceptions, make them specific — derive from `std::runtime_error`, not `std::exception`.
- Use `noexcept` on functions that cannot throw (destructors, move operations).

---

# Rust-Specific Rules

## 25. Embrace the Ownership Model

- Don't fight the borrow checker — redesign your data flow instead.
- Prefer passing references (`&T`, `&mut T`) over cloning. Clone only when ownership transfer is genuinely needed.
- Use lifetimes explicitly only when the compiler can't infer them — don't annotate unnecessarily.
- Prefer `&str` over `String` in function parameters when you don't need ownership.

## 26. Use Iterators and Combinators

- Prefer iterator chains (`.iter().filter().map().collect()`) over manual loops with indices.
- Use `for item in &collection` instead of `for i in 0..collection.len()`.
- Use `enumerate()`, `zip()`, `chain()`, `chunks()` — the iterator API is rich.
- Avoid `.unwrap()` in production code — use `?`, `unwrap_or`, `unwrap_or_else`, or pattern matching.

```rust
// Bad: manual indexing
let mut names = Vec::new();
for i in 0..users.len() {
    if users[i].is_active {
        names.push(users[i].name.clone());
    }
}

// Good: idiomatic iterator chain
let names: Vec<_> = users.iter()
    .filter(|u| u.is_active)
    .map(|u| u.name.clone())
    .collect();
```

## 27. Use Enums and Pattern Matching

- Use `enum` with data variants instead of class hierarchies or tagged unions.
- Use `match` exhaustively — the compiler ensures you handle all cases.
- Use `if let` / `while let` for single-variant matching instead of full `match`.
- Prefer `Result<T, E>` over panicking — make errors part of the type signature.

## 28. Leverage the Type System

- Use **newtype wrappers** (`struct UserId(u64)`) to prevent mixing up same-typed values.
- Use `Option<T>` instead of sentinel values or null pointers.
- Use `#[must_use]` on functions whose return values shouldn't be ignored.
- Prefer `From`/`Into` traits for type conversions over manual conversion functions.

## 29. Module Organization

- Keep `pub` surfaces small — expose only what's needed.
- Use `pub(crate)` for crate-internal visibility instead of full `pub`.
- Group related types and functions in modules — one concept per module.
- Re-export key types at the crate root for ergonomic imports.

---

# CUDA-Specific Rules

## 30. Name Kernels and Device Functions Clearly

- Kernel names should describe **what** they compute, not that they're kernels: `reduce_sum` not `kernel1` or `myKernel`.
- Use a consistent naming convention to distinguish execution spaces: e.g., `reduce_sum_kernel` for `__global__`, `warp_reduce` for `__device__` helpers.
- Name grid/block dimension variables descriptively: `threads_per_block`, `num_blocks` not `tpb`, `nb`, or bare `256`.

```cuda
// Bad: opaque names, magic numbers
__global__ void k1(float *a, float *b, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) b[i] = a[i] * 2.0f;
}
k1<<<(n+255)/256, 256>>>(d_in, d_out, n);

// Good: clear intent, named constants
constexpr int THREADS_PER_BLOCK = 256;

__global__ void scale_kernel(const float *input, float *output,
                             float scale_factor, int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = input[idx] * scale_factor;
    }
}

const int num_blocks = (num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
scale_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, 2.0f, num_elements);
```

## 31. Separate Host Logic from Device Logic

- Keep **host orchestration** (memory allocation, transfers, kernel launches, synchronization) in separate functions from **device computation** (kernels and device helpers).
- Don't mix `cudaMalloc`/`cudaMemcpy` with application logic — wrap them in RAII classes or helper functions.
- Use a clear file organization: consider separating `.cu` kernel files from `.cpp` host logic files, or at minimum group host and device code into clearly labeled sections.

```cpp
// Good: RAII wrapper hides allocation/deallocation
template <typename T>
class DeviceBuffer {
    T *ptr_ = nullptr;
    size_t size_ = 0;
public:
    explicit DeviceBuffer(size_t count) : size_(count) {
        check_cuda(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) { o.ptr_ = nullptr; }

    T *get() { return ptr_; }
    const T *get() const { return ptr_; }
    size_t size() const { return size_; }

    void copy_from_host(const T *host_data) {
        check_cuda(cudaMemcpy(ptr_, host_data, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copy_to_host(T *host_data) const {
        check_cuda(cudaMemcpy(host_data, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
};
```

## 32. Always Check CUDA Errors

- **Check every CUDA API call.** Silent failures are the #1 source of hard-to-debug CUDA issues.
- Use a `check_cuda` macro or inline function — not raw `if` blocks after every call.
- Check errors after kernel launches with `cudaGetLastError()` + `cudaDeviceSynchronize()` during development.
- In release builds, at minimum check allocations and memcpy — these are the most likely to fail at runtime.

```cuda
// Good: concise, catches file/line info
inline void check_cuda(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d — %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define check_cuda(err) check_cuda((err), __FILE__, __LINE__)

// Usage
check_cuda(cudaMalloc(&d_ptr, size));
my_kernel<<<grid, block>>>(d_ptr, n);
check_cuda(cudaGetLastError());
check_cuda(cudaDeviceSynchronize());
```

## 33. Make Thread Indexing Obvious

- Compute the global thread index **once** at the top of the kernel and store it in a clearly named variable.
- Use **early return** for out-of-bounds threads — don't wrap the entire kernel body in an `if`.
- For 2D/3D grids, name dimensions explicitly: `row`, `col`, `depth` — not `x`, `y`, `z`.

```cuda
// Bad: index computed inline, entire body wrapped
__global__ void process(float *data, int width, int height) {
    if (blockIdx.x * blockDim.x + threadIdx.x < width &&
        blockIdx.y * blockDim.y + threadIdx.y < height) {
        int idx = (blockIdx.y * blockDim.y + threadIdx.y) * width +
                  (blockIdx.x * blockDim.x + threadIdx.x);
        data[idx] = data[idx] * 2.0f;
    }
}

// Good: named indices, early return
__global__ void process(float *data, int width, int height) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    const int idx = row * width + col;
    data[idx] = data[idx] * 2.0f;
}
```

## 34. Document Shared Memory Usage

- Declare shared memory with a **descriptive name** that indicates what it holds: `shared_tile` not `smem` or `s`.
- Add a brief comment explaining the **size** and **purpose** of shared memory when it's dynamically allocated (`extern __shared__`).
- Keep the shared memory lifecycle short — load, sync, compute, sync — and make each phase visually distinct.

```cuda
// Good: clear phases, descriptive names
__global__ void tiled_matmul_kernel(const float *A, const float *B,
                                     float *C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float accumulator = 0.0f;

    for (int tile_idx = 0; tile_idx < N / TILE_SIZE; ++tile_idx) {
        // Phase 1: Load tiles from global memory
        tile_A[threadIdx.y][threadIdx.x] = A[row * N + tile_idx * TILE_SIZE + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(tile_idx * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // Phase 2: Compute partial dot product from tiles
        for (int k = 0; k < TILE_SIZE; ++k) {
            accumulator += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = accumulator;
}
```

## 35. Keep Kernels Short — Extract Device Helpers

- Apply the same "one function, one task" rule to kernels. If a kernel does loading, computing, and reducing, extract `__device__` helper functions.
- Use `__forceinline__ __device__` for small helpers that you want inlined without relying on compiler heuristics.
- This makes kernels easier to read, test (via unit-testing device functions), and reuse.

```cuda
// Good: kernel reads like pseudocode, details in helpers
__forceinline__ __device__
float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__forceinline__ __device__
float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

__global__ void reduce_sum_kernel(const float *input, float *output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float val = (idx < n) ? input[idx] : 0.0f;

    const float block_sum = block_reduce_sum(val);
    if (threadIdx.x == 0) atomicAdd(output, block_sum);
}
```

## 36. Be Explicit About Memory Spaces

- Use `const` on kernel parameters for read-only device pointers — documents intent and enables compiler optimizations.
- Use `__restrict__` when pointers don't alias — but add a comment explaining the non-aliasing guarantee.
- When using unified memory (`cudaMallocManaged`), comment the expected access pattern (host-only init, device-only compute, etc.) — the implicit page migration behavior is not obvious.
- Prefer explicit memory copies over unified memory in performance-critical paths — be explicit about data movement.

```cuda
// Good: const + restrict with clear intent
__global__ void vector_add_kernel(
    const float *__restrict__ a,   // read-only, no alias with output
    const float *__restrict__ b,   // read-only, no alias with output
    float *__restrict__ output,    // write-only
    int num_elements)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    output[idx] = a[idx] + b[idx];
}
```

## 37. Synchronization: Make It Visible and Minimal

- Place `__syncthreads()` on its own line, never buried inside a conditional branch that not all threads take — this is **undefined behavior** and hard to spot.
- Add a brief comment before each `__syncthreads()` stating what invariant it establishes: "all threads have loaded their tile", "partial sums are written to shared memory".
- Minimize synchronization points — restructure algorithms to reduce the number of barriers.
- For warp-level operations, prefer warp intrinsics (`__shfl_sync`, `__ballot_sync`) with explicit masks over `__syncthreads()`.

## 38. Launch Configuration: Make It Readable

- Wrap kernel launches in a **host function** that computes and names the launch parameters.
- Never hardcode grid/block dimensions at the call site — compute them from the problem size.
- For complex launch configurations, use a struct or helper to make the 2D/3D grid/block shape clear.
- Query device properties at startup rather than assuming specific hardware limits.

```cuda
// Bad: magic numbers, unclear intent
foo<<<(n+127)/128, 128, 0, stream>>>(d_ptr, n);

// Good: named, computed, self-documenting
void launch_scale_kernel(float *d_data, float factor, int n, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(d_data, factor, n);
    check_cuda(cudaGetLastError());
}
```

## 39. Streams and Async: Comment the Dependency Graph

- When using multiple CUDA streams, add a comment block showing the **dependency graph** — which operations must complete before others begin.
- Name streams after their purpose: `compute_stream`, `transfer_stream` — not `s1`, `s2`.
- Group related async operations visually and separate independent pipelines with blank lines.
- Always synchronize before reading results on the host — make the sync point explicit and commented.

```cuda
// Good: dependency graph documented, streams named by purpose
// Dependency graph:
//   upload (transfer_stream) --> compute (compute_stream) --> download (transfer_stream)
//   Event 'upload_done' gates compute start.
//   Event 'compute_done' gates download start.

cudaStream_t transfer_stream, compute_stream;
cudaEvent_t upload_done, compute_done;

// Stage 1: async upload
cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, transfer_stream);
cudaEventRecord(upload_done, transfer_stream);

// Stage 2: compute waits for upload
cudaStreamWaitEvent(compute_stream, upload_done);
process_kernel<<<grid, block, 0, compute_stream>>>(d_input, d_output, n);
cudaEventRecord(compute_done, compute_stream);

// Stage 3: download waits for compute
cudaStreamWaitEvent(transfer_stream, compute_done);
cudaMemcpyAsync(h_output, d_output, size, cudaMemcpyDeviceToHost, transfer_stream);

// Sync before host reads the result
cudaStreamSynchronize(transfer_stream);
```
