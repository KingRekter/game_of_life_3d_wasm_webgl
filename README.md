# 3D WebAssembly Implementation of Conway's Game of Life

This project is an advanced implementation of Conway's Game of Life, developed in C++ and compiled to WebAssembly (WASM) for high-performance execution in modern web browsers. It leverages WebGL for 3D hardware-accelerated rendering of a massive 1920x1080 grid, capable of simulating and visualizing hundreds of thousands of cells in real-time.

## Technical Overview

- **Core Logic:** C++17
- **Compilation Target:** WebAssembly (WASM)
- **Rendering API:** WebGL 2.0 via OpenGL ES 3.0 bindings
- **Grid Dimensions:** 1920x1080
- **UI/Overlay:** Dear ImGui

## Optimization Strategies

The primary challenge in this project was achieving real-time performance for a large-scale simulation and rendering task within a browser environment. This was accomplished through several key optimization techniques:

### 1. GPU-Accelerated Instanced Rendering

To render a vast number of cells without overwhelming the CPU with individual draw calls, this implementation uses **instanced rendering**. The geometry for a single cube is uploaded to the GPU once. Then, for each frame, an array of transformation and color data for every visible cell is sent to the GPU. A single `glDrawArraysInstanced` call then renders all cells, offloading the bulk of the transformation work to the GPU. This method significantly reduces CPU-to-GPU communication overhead, a common bottleneck in graphics-intensive applications.

### 2. Active Region Processing for Simulation

A naive simulation approach would iterate over every cell in the 1920x1080 grid, which is computationally expensive. This implementation employs a sparse grid optimization by processing only an **"active region."** In each generation, the simulation logic considers only the cells that were alive in the previous state and their immediate neighbors. This approach is highly effective because in typical Game of Life patterns, the number of active cells is a small fraction of the total grid area, drastically reducing the computational load.

### 3. Data-Oriented Design for Cache Efficiency

Instead of a conventional 2D array, which would be sparsely populated and memory-inefficient, the application utilizes a **data-oriented design**. Live cells are stored as a packed `std::vector<CellData>`. This contiguous memory layout ensures high cache coherence. When the CPU processes the list of active cells, the required data is more likely to be present in the cache, minimizing costly main memory access and improving the performance of the simulation loop.

### 4. Spatial Partitioning Framework

The codebase includes a `spatialGrid` structure, which partitions the grid into chunks. While the current single-threaded simulation does not fully exploit this for neighbor lookups, it establishes a framework for future scalability. This design would allow a multi-threaded or compute-shader-based implementation to process grid chunks in parallel, further optimizing the simulation by localizing memory access and reducing data contention.

### 5. Static Buffer Allocation

To avoid performance stalls associated with dynamic memory allocation during the render loop, memory for render data buffers is pre-allocated to a maximum capacity at startup. This ensures that the application does not incur the overhead of heap allocation (`new`/`delete`) or vector resizing during real-time operation, leading to smoother and more predictable frame rates.

## Build and Execution

Building the project requires the Emscripten SDK.

1.  **Clone the repository and navigate to the project directory.**

2.  **Compile the C++ source to WASM and HTML:**
    ```bash
    emcc -o gameoflife3d.html main.cpp -s USE_GLFW=3 -s USE_WEBGL2=1 -s FULL_ES3=1 -s ALLOW_MEMORY_GROWTH=1 -s ASYNCIFY -O3 -std=c++17 -Ilib
    ```

3.  **Launch a local web server:**
    ```bash
    python3 -m http.server
    ```

4.  **Access the application** by navigating to `http://localhost:8000/gameoflife3d.html` in your browser.

## Controls

*   **Camera Rotation:** Left-click and drag, or use WASD/Arrow Keys.
*   **Camera Pan:** Middle-click and drag, or use I/J/K/L keys.
*   **Camera Zoom:** Mouse scroll wheel, or use Q/E keys.
*   **Simulation:**
    *   `Spacebar`: Pause/Resume
    *   `R`: Reset with a new random pattern
    *   `+/-`: Adjust simulation speed
*   **Display:**
    *   `H`: Toggle statistics overlay
    *   `T`: Toggle auto-rotation
    *   `C`: Cycle color schemes
    *   `[` / `]`: Adjust color spread

## Dependencies

- [GLFW](https://www.glfw.org/) (as part of Emscripten ports)
- [GLM](https://glm.g-truc.net/0.9.9/index.html)
- [Dear ImGui](https://github.com/ocornut/imgui)
- [Emscripten SDK](https://emscripten.org/)