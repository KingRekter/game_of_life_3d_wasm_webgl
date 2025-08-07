#include <GLES3/gl3.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <array>
#include <map>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

// --- Configuration ---
const int WINDOW_WIDTH = 1920;
const int WINDOW_HEIGHT = 1080;
const int GRID_WIDTH = 1920;
const int GRID_HEIGHT = 1080;
const float VOXEL_SIZE = 0.015f;

// Grid system for efficient updates
const int GRID_SIZE = 32;
const int CHUNK_SIZE = GRID_WIDTH / GRID_SIZE;
std::vector<std::vector<std::vector<int>>> spatialGrid(GRID_SIZE, std::vector<std::vector<int>>(GRID_SIZE));
static std::vector<std::atomic<char>> activeFlagsGrid(GRID_WIDTH * GRID_HEIGHT);

// --- Global State ---
unsigned int updatesPerSecond = 20;
bool isPaused = false;
bool showOverlay = false;
float g_hueOffset = 0.0f;
float g_colorSpread = 1.0f;

// --- Camera and Input State with PANNING ---
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 8.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 panOffset = glm::vec3(0.0f, 0.0f, 0.0f);
float rotationX = 90.0f;
float rotationY = 0.0f;
float zoomLevel = 1.9f;
float rotationSpeed = 50.0f;
float panSpeed = 15.0f;
bool autoRotate = false;
GLFWwindow *g_window = nullptr;
float lastX = WINDOW_WIDTH / 2.0f;
float lastY = WINDOW_HEIGHT / 2.0f;
bool firstMouse = true;
bool mousePressed = false;
bool middleMousePressed = false;
bool rightMousePressed = false;

// --- Random Number Generation ---
std::random_device rd;
std::mt19937 rng(rd());

// --- Data Structures ---
struct InstanceData
{
    glm::vec3 position;
    glm::vec4 color;
    bool isDying = false;
};

struct CellData {
    int x;
    int y;
    uint8_t neighbors;
    bool isNewBorn;
};

// --- CORE GAME STATE ---
std::vector<bool> currentGrid(GRID_WIDTH * GRID_HEIGHT, false);
std::vector<bool> nextGrid(GRID_WIDTH * GRID_HEIGHT, false);
std::vector<CellData> aliveCellsData;

// --- Thread-safe Rendering ---
struct RenderData
{
    std::vector<InstanceData> instances;
    size_t instanceCount = 0;
    bool needsUpdate = false;
};
RenderData renderBuffers[2];
std::mutex renderDataMutex;
const size_t MAX_INSTANCES = 800000;

// --- OpenGL and Stats ---
GLuint shaderProgram;
GLuint cubeVAO, cubeVBO, instanceVBO;
GLuint boundaryVAO, boundaryVBO, boundaryShaderProgram;

// Simple overlay rendering
GLuint overlayShaderProgram;
GLuint overlayVAO, overlayVBO;

struct UniformCache
{
    GLint model, view, projection, lightPos, viewPos;
    void init(GLuint program)
    {
        model = glGetUniformLocation(program, "model");
        view = glGetUniformLocation(program, "view");
        projection = glGetUniformLocation(program, "projection");
        lightPos = glGetUniformLocation(program, "lightPos");
        viewPos = glGetUniformLocation(program, "viewPos");
    }
} uniforms;

struct GameStats
{
    size_t totalCells = 0;
    size_t generation = 0;
    float fps = 0.0f;
    double updateTime = 0.0;
} gameStats;

glm::vec3 hslToRgb(float h, float s, float l) {
    float r, g, b;
    if (s == 0.0f) {
        r = g = b = l;
    } else {
        auto hue2rgb = [](float p, float q, float t) {
            if (t < 0.0f) t += 1.0f;
            if (t > 1.0f) t -= 1.0f;
            if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
            if (t < 1.0f / 2.0f) return q;
            if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
            return p;
        };
        float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
        float p = 2.0f * l - q;
        r = hue2rgb(p, q, h + 1.0f / 3.0f);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1.0f / 3.0f);
    }
    return glm::vec3(r, g, b);
}

// --- Helper Functions ---
inline bool isValidCell(int x, int y)
{
    return x >= 0 && x < GRID_WIDTH && y >= 0 && y < GRID_HEIGHT;
}

inline int getGridIndex(int x, int y)
{
    return x + y * GRID_WIDTH;
}

inline int getSpatialGridIndex(int x, int y)
{
    return (x / CHUNK_SIZE) + (y / CHUNK_SIZE) * GRID_SIZE;
}

inline glm::vec4 getCellColor(int neighbors, bool isNewBorn)
{
    if (isNewBorn && g_colorSpread < 0.25f) {
        return glm::vec4(1.0f, 1.0f, 1.0f, 0.9f);
    }

    const float targetHue = 120.0f / 360.0f;
    float fullSpreadHue = 0.0f / 360.0f;
    if (neighbors == 2) {
        fullSpreadHue = 240.0f / 360.0f;
    } else if (neighbors == 3) {
        fullSpreadHue = 120.0f / 360.0f;
    }

    float baseHue = targetHue + (fullSpreadHue - targetHue) * g_colorSpread;
    float saturation = 0.95f;
    float lightness = 0.5f;
    float finalHue = fmod(baseHue + (g_hueOffset / 360.0f), 1.0f);
    if (finalHue < 0.0f) finalHue += 1.0f;

    glm::vec3 rgbColor = hslToRgb(finalHue, saturation, lightness);
    return glm::vec4(rgbColor, 0.85f);
}

void initializeRandomPattern(float density = 0.2f)
{
    std::cout << "Initializing FULL GRID pattern with density: " << density << std::endl;
    
    std::fill(currentGrid.begin(), currentGrid.end(), false);
    std::fill(nextGrid.begin(), nextGrid.end(), false);
    aliveCellsData.clear();
    
    for (auto& row : spatialGrid) {
        for (auto& chunk : row) {
            chunk.clear();
        }
    }
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int y = 0; y < GRID_HEIGHT; y++)
    {
        for (int x = 0; x < GRID_WIDTH; x++)
        {
            if (dist(rng) < density)
            {
                int idx = getGridIndex(x, y);
                currentGrid[idx] = true;
                aliveCellsData.push_back({x, y, 0, true}); 
                
                int gridX = x / CHUNK_SIZE;
                int gridY = y / CHUNK_SIZE;
                if (gridX >= 0 && gridX < GRID_SIZE && gridY >= 0 && gridY < GRID_SIZE) {
                    spatialGrid[gridY][gridX].push_back(idx);
                }
            }
        }
    }
    
    gameStats.totalCells = aliveCellsData.size();
    gameStats.generation = 0;
    
    std::cout << "Initialized FULL GRID with " << gameStats.totalCells 
              << " cells across entire " << GRID_WIDTH << "x" << GRID_HEIGHT << " grid!" << std::endl;
}

// Simplified single-threaded update for WebGL compatibility
void updateMassive2DGameOfLife()
{
    if (isPaused) return;

    double startTime = glfwGetTime();
    
    // Get all active regions (cells + neighbors) - single threaded
    std::vector<bool> activeRegion(GRID_WIDTH * GRID_HEIGHT, false);
    
    for (const auto& cell : aliveCellsData) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cell.x + dx;
                int ny = cell.y + dy;
                if (isValidCell(nx, ny)) {
                    activeRegion[getGridIndex(nx, ny)] = true;
                }
            }
        }
    }
    
    std::vector<CellData> nextAliveCellsData;
    
    // Calculate next generation - single threaded
    for (size_t idx = 0; idx < GRID_WIDTH * GRID_HEIGHT; idx++) {
        if (!activeRegion[idx]) continue;
            
        int x = idx % GRID_WIDTH;
        int y = idx / GRID_WIDTH;
        
        int neighbors = 0;
        if (x > 0 && y > 0) neighbors += currentGrid[getGridIndex(x-1, y-1)];
        if (y > 0) neighbors += currentGrid[getGridIndex(x, y-1)];
        if (x < GRID_WIDTH-1 && y > 0) neighbors += currentGrid[getGridIndex(x+1, y-1)];
        if (x > 0) neighbors += currentGrid[getGridIndex(x-1, y)];
        if (x < GRID_WIDTH-1) neighbors += currentGrid[getGridIndex(x+1, y)];
        if (x > 0 && y < GRID_HEIGHT-1) neighbors += currentGrid[getGridIndex(x-1, y+1)];
        if (y < GRID_HEIGHT-1) neighbors += currentGrid[getGridIndex(x, y+1)];
        if (x < GRID_WIDTH-1 && y < GRID_HEIGHT-1) neighbors += currentGrid[getGridIndex(x+1, y+1)];
        
        bool isAlive = currentGrid[idx];
        bool willLive = isAlive ? (neighbors == 2 || neighbors == 3) : (neighbors == 3);
        
        if (willLive) {
            bool isNewBorn = !isAlive;
            nextAliveCellsData.push_back({x, y, (uint8_t)neighbors, isNewBorn});
        }
    }
    
    // Update grid state
    std::fill(nextGrid.begin(), nextGrid.end(), false);
    for(const auto& cell : nextAliveCellsData) {
        nextGrid[getGridIndex(cell.x, cell.y)] = true;
    }

    currentGrid.swap(nextGrid);
    aliveCellsData = std::move(nextAliveCellsData);
    
    // Rebuild spatial grid
    for (auto& row : spatialGrid) { for (auto& chunk : row) { chunk.clear(); } }
    for (const auto& cell : aliveCellsData) {
        int gridX = cell.x / CHUNK_SIZE;
        int gridY = cell.y / CHUNK_SIZE;
        if (gridX >= 0 && gridX < GRID_SIZE && gridY >= 0 && gridY < GRID_SIZE) {
            spatialGrid[gridY][gridX].push_back(getGridIndex(cell.x, cell.y));
        }
    }
    
    gameStats.totalCells = aliveCellsData.size();
    gameStats.generation++;
    gameStats.updateTime = (glfwGetTime() - startTime) * 1000.0;
    
    // Create render data
    size_t renderLimit = std::min(gameStats.totalCells, MAX_INSTANCES);
    std::vector<InstanceData> renderData;
    renderData.reserve(renderLimit);
    
    for (size_t i = 0; i < renderLimit; i++) {
        const auto& cell = aliveCellsData[i];
        
        glm::vec3 position(
            (cell.x - GRID_WIDTH / 2.0f) * VOXEL_SIZE,
            0.0f,
            (cell.y - GRID_HEIGHT / 2.0f) * VOXEL_SIZE
        );
        
        renderData.push_back({position, getCellColor(cell.neighbors, cell.isNewBorn), false});
    }
    
    {
        std::lock_guard<std::mutex> lock(renderDataMutex);
        renderBuffers[0].instances = std::move(renderData);
        renderBuffers[0].instanceCount = renderBuffers[0].instances.size();
        renderBuffers[0].needsUpdate = true;
    }
    
    if (gameStats.generation % 10 == 0) {
        std::cout << "Gen " << gameStats.generation << ": " 
                  << gameStats.totalCells << " cells, "
                  << std::fixed << std::setprecision(1) << gameStats.updateTime << "ms, "
                  << renderLimit << " rendered" << std::endl;
    }
}

std::atomic<bool> isInitializing = false;
std::atomic<bool> shouldReset = false;
std::mutex fullSystemMutex;

void initializeMassive2DPattern()
{
    std::cout << "Starting massive 2D initialization..." << std::endl;
    shouldReset.store(true);
    isInitializing.store(true);
    std::lock_guard<std::mutex> systemLock(fullSystemMutex);
    
    initializeRandomPattern(0.1f);
    
    {
        std::lock_guard<std::mutex> renderLock(renderDataMutex);
        renderBuffers[0].instances.clear();
        
        size_t renderLimit = std::min(gameStats.totalCells, MAX_INSTANCES);
        renderBuffers[0].instances.reserve(renderLimit);
        
        for (size_t i = 0; i < renderLimit; i++) {
            glm::vec3 position(
                (aliveCellsData[i].x - GRID_WIDTH / 2.0f) * VOXEL_SIZE,
                0.0f,
                (aliveCellsData[i].y - GRID_HEIGHT / 2.0f) * VOXEL_SIZE
            );
            
            renderBuffers[0].instances.push_back({
                position,
                glm::vec4(0.4f, 0.8f, 0.4f, 0.8f),
                false
            });
        }
        
        renderBuffers[0].instanceCount = renderBuffers[0].instances.size();
        renderBuffers[0].needsUpdate = true;
    }
    
    shouldReset.store(false);
    isInitializing.store(false);
    std::cout << "Massive 2D initialization complete." << std::endl;
}

// --- Input and Main Loop with PANNING ---
void processInput(GLFWwindow *window, float deltaTime)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
        
    float rotSpeed = rotationSpeed * deltaTime;
    float zoomSpeed = 3.0f * deltaTime;
    float currentPanSpeed = panSpeed * deltaTime * zoomLevel;
    
    // Rotation controls
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        rotationY -= rotSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        rotationY += rotSpeed;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        rotationX -= rotSpeed;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        rotationX += rotSpeed;
    
    // PANNING CONTROLS
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
        panOffset.z += currentPanSpeed;
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        panOffset.z -= currentPanSpeed;
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
        panOffset.x -= currentPanSpeed;
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
        panOffset.x += currentPanSpeed;
    
    // Numpad panning
    if (glfwGetKey(window, GLFW_KEY_KP_8) == GLFW_PRESS)
        panOffset.z += currentPanSpeed;
    if (glfwGetKey(window, GLFW_KEY_KP_2) == GLFW_PRESS)
        panOffset.z -= currentPanSpeed;
    if (glfwGetKey(window, GLFW_KEY_KP_4) == GLFW_PRESS)
        panOffset.x -= currentPanSpeed;
    if (glfwGetKey(window, GLFW_KEY_KP_6) == GLFW_PRESS)
        panOffset.x += currentPanSpeed;
    
    // Reset pan
    if (glfwGetKey(window, GLFW_KEY_HOME) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_5) == GLFW_PRESS)
        panOffset = glm::vec3(0.0f, 0.0f, 0.0f);
    
    // Zoom controls
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        zoomLevel *= (1.0f + zoomSpeed);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        zoomLevel *= (1.0f - zoomSpeed);
        
    zoomLevel = std::max(0.5f, std::min(20.0f, zoomLevel));
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS) return;
    
    switch (key)
    {
    case GLFW_KEY_LEFT_BRACKET:
        g_colorSpread = std::max(0.0f, g_colorSpread - 0.05f);
        break;
    case GLFW_KEY_RIGHT_BRACKET:
        g_colorSpread = std::min(1.0f, g_colorSpread + 0.05f);
        break;
    case GLFW_KEY_C:
        g_hueOffset += 30.0f;
        if (g_hueOffset >= 360.0f) g_hueOffset -= 360.0f;
        break;
    case GLFW_KEY_SPACE:
        isPaused = !isPaused;
        break;
    case GLFW_KEY_R:
        initializeMassive2DPattern();
        break;
    case GLFW_KEY_T:
        autoRotate = !autoRotate;
        break;
    case GLFW_KEY_H:
        showOverlay = !showOverlay;
        break;
    case GLFW_KEY_KP_ADD:
    case GLFW_KEY_EQUAL:
        updatesPerSecond = std::min(60u, updatesPerSecond + 1u);
        break;
    case GLFW_KEY_KP_SUBTRACT:
    case GLFW_KEY_MINUS:
        updatesPerSecond = std::max(1u, updatesPerSecond - 1u);
        break;
    case GLFW_KEY_1:
        initializeRandomPattern(0.15f);
        break;
    case GLFW_KEY_2:
        initializeRandomPattern(0.22f);
        break;
    case GLFW_KEY_3:
        initializeRandomPattern(0.25f);
        break;
    case GLFW_KEY_4:
        initializeRandomPattern(0.3f);
        break;
    }
}

void mouseCallback(GLFWwindow *window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = static_cast<float>(xpos);
        lastY = static_cast<float>(ypos);
        firstMouse = false;
    }
    
    float xoffset = static_cast<float>(xpos - lastX);
    float yoffset = static_cast<float>(lastY - ypos);
    
    if (mousePressed)
    {
        rotationY += xoffset * 0.1f;
        rotationX += yoffset * 0.1f;
    }
    
    if (middleMousePressed || rightMousePressed)
    {
        float panSensitivity = 0.01f * zoomLevel;
        float radY = glm::radians(rotationY);
        glm::vec3 right = glm::vec3(cos(radY), 0, sin(radY));
        glm::vec3 forward = glm::vec3(-sin(radY), 0, cos(radY));
        
        panOffset += right * (-xoffset * panSensitivity);
        panOffset += forward * (yoffset * panSensitivity);
    }
    
    lastX = static_cast<float>(xpos);
    lastY = static_cast<float>(ypos);
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        mousePressed = (action == GLFW_PRESS);
        if (action == GLFW_PRESS) autoRotate = false;
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
    {
        middleMousePressed = (action == GLFW_PRESS);
        if (action == GLFW_PRESS) autoRotate = false;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        rightMousePressed = (action == GLFW_PRESS);
        if (action == GLFW_PRESS) autoRotate = false;
    }
}

void scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    zoomLevel *= (1.0f - static_cast<float>(yoffset) * 0.1f);
    zoomLevel = std::max(0.5f, std::min(20.0f, zoomLevel));
}

// Add this function before renderOverlay()
void drawSimpleChar(char c, float x, float y, float size) {
    std::vector<float> lines;
    
    switch(c) {
        case '0':
            lines = {x,y, x+size,y, x+size,y, x+size,y+size, x+size,y+size, x,y+size, x,y+size, x,y};
            break;
        case '1':
            lines = {x+size/2,y, x+size/2,y+size};
            break;
        case '2':
            lines = {x,y+size, x+size,y+size, x+size,y+size, x+size,y+size/2, x+size,y+size/2, x,y+size/2, x,y+size/2, x,y, x,y, x+size,y};
            break;
        case '3':
            lines = {x,y, x+size,y, x+size,y, x+size,y+size/2, x+size,y+size/2, x,y+size/2, x+size,y+size/2, x+size,y+size, x+size,y+size, x,y+size};
            break;
        case 'G': case 'g':
            lines = {x,y+size, x+size,y+size, x+size,y+size, x+size,y+size/2, x+size,y+size/2, x+size/2,y+size/2, x,y+size, x,y, x,y, x+size,y};
            break;
        case 'e':
            lines = {x,y, x+size,y, x+size,y, x+size,y+size/2, x+size,y+size/2, x,y+size/2, x,y+size/2, x,y+size, x,y+size, x+size,y+size};
            break;
        case 'n':
            lines = {x,y, x,y+size, x,y+size/2, x+size,y+size, x+size,y+size/2, x+size,y};
            break;
        case ':':
            lines = {x+size/2,y+size/4, x+size/2,y+size/4, x+size/2,y+3*size/4, x+size/2,y+3*size/4};
            break;
        case ' ':
            break; // No lines for space
        default:
            // Draw a simple rectangle for unknown chars
            lines = {x,y, x+size,y, x+size,y, x+size,y+size, x+size,y+size, x,y+size, x,y+size, x,y};
            break;
    }
    
    if (!lines.empty()) {
        glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINES, 0, lines.size()/2);
    }
}

void drawSimpleText(const std::string& text, float x, float y, float charSize) {
    float currentX = x;
    for (char c : text) {
        drawSimpleChar(c, currentX, y, charSize);
        currentX += charSize + 2.0f; // Add spacing between characters
    }
}

void renderOverlay()
{
    // Window title (always shown)
    std::stringstream titleStream;
    titleStream << "Conway's Game of Life - Generation: " << gameStats.generation;
    if (isPaused) titleStream << " [PAUSED]";
    glfwSetWindowTitle(g_window, titleStream.str().c_str());
    
    if (showOverlay) {
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create stats window
        ImGui::Begin("Game Statistics", &showOverlay);
        ImGui::Text("Generation: %zu", gameStats.generation);
        ImGui::Text("Live Cells: %zu", gameStats.totalCells);
        ImGui::Text("Speed: %u ups", updatesPerSecond);
        ImGui::Text("FPS: %.0f", gameStats.fps);
        ImGui::Text("Update: %.1fms", gameStats.updateTime);
        ImGui::Text("Zoom: %.1fx", zoomLevel);
        ImGui::Text("Pan: (%.1f, %.1f)", panOffset.x, panOffset.z);
        ImGui::Text("Status: %s", isPaused ? "PAUSED" : "RUNNING");
        ImGui::End();

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
}
// Fixed shader sources
const char *vertexShaderSource = "#version 300 es\nlayout (location = 0) in vec3 aPos;\nlayout (location = 1) in vec3 aInstancePos;\nlayout (location = 2) in vec4 aInstanceColor;\nuniform mat4 model;\nuniform mat4 view;\nuniform mat4 projection;\nout vec4 FragColor;\nout vec3 FragPos;\nout vec3 Norm;\nvoid main()\n{\n    Norm = normalize(aPos);\n    vec3 worldPos = aPos + aInstancePos;\n    FragPos = vec3(model * vec4(worldPos, 1.0));\n    FragColor = aInstanceColor;\n    gl_Position = projection * view * model * vec4(worldPos, 1.0);\n}\n";

const char *fragmentShaderSource = "#version 300 es\nprecision mediump float;\nin vec4 FragColor;\nin vec3 FragPos;\nin vec3 Norm;\nout vec4 finalColor;\nuniform vec3 lightPos;\nuniform vec3 viewPos;\nvoid main()\n{\n    vec3 ambient = 0.7 * FragColor.rgb;\n    vec3 norm = normalize(Norm);\n    vec3 lightDir = normalize(lightPos - FragPos);\n    float diff = max(dot(norm, lightDir), 0.0);\n    vec3 diffuse = diff * FragColor.rgb * 0.3;\n    vec3 result = ambient + diffuse;\n    finalColor = vec4(result, FragColor.a);\n}\n";

const char *boundaryVertexShaderSource = "#version 300 es\nlayout (location = 0) in vec3 aPos;\nuniform mat4 model;\nuniform mat4 view;\nuniform mat4 projection;\nvoid main()\n{\n    gl_Position = projection * view * model * vec4(aPos, 1.0);\n}\n";

const char *boundaryFragmentShaderSource = "#version 300 es\nprecision mediump float;\nout vec4 finalColor;\nvoid main()\n{\n    finalColor = vec4(0.1, 0.1, 0.4, 0.5);\n}\n";

// Simple overlay shaders
const char *overlayVertexShaderSource = "#version 300 es\nlayout (location = 0) in vec2 aPos;\nuniform mat4 projection;\nvoid main()\n{\n    gl_Position = projection * vec4(aPos, 0.0, 1.0);\n}\n";

const char *overlayFragmentShaderSource = "#version 300 es\nprecision mediump float;\nout vec4 finalColor;\nuniform vec4 color;\nvoid main()\n{\n    finalColor = color;\n}\n";

GLuint compileShader(const char *source, GLenum type)
{
   GLuint shader = glCreateShader(type);
   glShaderSource(shader, 1, &source, NULL);
   glCompileShader(shader);
   GLint success;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
   if (!success)
   {
       char infoLog[512];
       glGetShaderInfoLog(shader, 512, NULL, infoLog);
       std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
   }
   return shader;
}

void initShaders()
{
   // Main 3D shaders
   GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
   GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
   shaderProgram = glCreateProgram();
   glAttachShader(shaderProgram, vertexShader);
   glAttachShader(shaderProgram, fragmentShader);
   glLinkProgram(shaderProgram);
   GLint success;
   glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
   if (!success)
   {
       char infoLog[512];
       glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
       std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
   }
   glDeleteShader(vertexShader);
   glDeleteShader(fragmentShader);

   // Boundary shaders
   GLuint boundaryVertexShader = compileShader(boundaryVertexShaderSource, GL_VERTEX_SHADER);
   GLuint boundaryFragmentShader = compileShader(boundaryFragmentShaderSource, GL_FRAGMENT_SHADER);
   boundaryShaderProgram = glCreateProgram();
   glAttachShader(boundaryShaderProgram, boundaryVertexShader);
   glAttachShader(boundaryShaderProgram, boundaryFragmentShader);
   glLinkProgram(boundaryShaderProgram);
   glDeleteShader(boundaryVertexShader);
   glDeleteShader(boundaryFragmentShader);

   // Overlay shaders
   GLuint overlayVertexShader = compileShader(overlayVertexShaderSource, GL_VERTEX_SHADER);
   GLuint overlayFragmentShader = compileShader(overlayFragmentShaderSource, GL_FRAGMENT_SHADER);
   overlayShaderProgram = glCreateProgram();
   glAttachShader(overlayShaderProgram, overlayVertexShader);
   glAttachShader(overlayShaderProgram, overlayFragmentShader);
   glLinkProgram(overlayShaderProgram);
   glGetProgramiv(overlayShaderProgram, GL_LINK_STATUS, &success);
   if (!success)
   {
       char infoLog[512];
       glGetProgramInfoLog(overlayShaderProgram, 512, NULL, infoLog);
       std::cerr << "ERROR::OVERLAY_SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
   }
   glDeleteShader(overlayVertexShader);
   glDeleteShader(overlayFragmentShader);
}

void render(GLFWwindow *window)
{
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   size_t instanceCount = 0;
   {
       std::lock_guard<std::mutex> lock(renderDataMutex);
       if (renderBuffers[0].needsUpdate)
       {
           instanceCount = renderBuffers[0].instanceCount;
           glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
           if (instanceCount > 0)
           {
               glBufferData(GL_ARRAY_BUFFER, renderBuffers[0].instances.size() * sizeof(InstanceData), 
                          renderBuffers[0].instances.data(), GL_STREAM_DRAW);
           }
           renderBuffers[0].needsUpdate = false;
       }
       else
       {
           instanceCount = renderBuffers[0].instanceCount;
       }
   }

   glm::mat4 model = glm::mat4(1.0f);
   model = glm::rotate(model, glm::radians(rotationX), glm::vec3(1.0f, 0.0f, 0.0f));
   model = glm::rotate(model, glm::radians(rotationY), glm::vec3(0.0f, 1.0f, 0.0f));
   
   glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f) + panOffset;
   glm::vec3 cameraPosition = (cameraPos * zoomLevel) + panOffset;
   
   glm::mat4 view = glm::lookAt(cameraPosition, cameraTarget, cameraUp);
   glm::mat4 projection = glm::perspective(glm::radians(45.0f), 
                                         (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, 0.1f, 200.0f);

   glUseProgram(shaderProgram);
   glUniformMatrix4fv(uniforms.model, 1, GL_FALSE, glm::value_ptr(model));
   glUniformMatrix4fv(uniforms.view, 1, GL_FALSE, glm::value_ptr(view));
   glUniformMatrix4fv(uniforms.projection, 1, GL_FALSE, glm::value_ptr(projection));
   glUniform3fv(uniforms.lightPos, 1, glm::value_ptr(glm::vec3(10.0f, 20.0f, 10.0f)));
   glUniform3fv(uniforms.viewPos, 1, glm::value_ptr(cameraPosition));

   if (instanceCount > 0)
   {
       glBindVertexArray(cubeVAO);
       glDrawArraysInstanced(GL_TRIANGLES, 0, 36, static_cast<GLsizei>(instanceCount));
   }

   // Draw boundary
   glUseProgram(boundaryShaderProgram);
   glUniformMatrix4fv(glGetUniformLocation(boundaryShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
   glUniformMatrix4fv(glGetUniformLocation(boundaryShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
   glUniformMatrix4fv(glGetUniformLocation(boundaryShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
   glBindVertexArray(boundaryVAO);
   glDrawArrays(GL_LINES, 0, 8);
   glBindVertexArray(0);

   // Render overlay last (on top of everything)
   renderOverlay();

   glfwSwapBuffers(window);
}

void mainLoopIteration()
{
   double currentFrameTime = glfwGetTime();
   static double lastFrameTime = 0.0;
   static double lastFpsTime = 0.0;
   static int frameCount = 0;
   static double lastUpdateTime = 0.0;

   double deltaTime = currentFrameTime - lastFrameTime;
   lastFrameTime = currentFrameTime;

   frameCount++;
   if (currentFrameTime - lastFpsTime >= 1.0)
   {
       gameStats.fps = static_cast<float>(frameCount);
       frameCount = 0;
       lastFpsTime = currentFrameTime;
   }

   processInput(g_window, static_cast<float>(deltaTime));
   if (autoRotate && !mousePressed)
       rotationY += 5.0f * static_cast<float>(deltaTime);

   // Update simulation at controlled rate
   if (currentFrameTime - lastUpdateTime >= (1.0 / updatesPerSecond))
   {
       updateMassive2DGameOfLife();
       lastUpdateTime = currentFrameTime;
   }

   render(g_window);
   glfwPollEvents();
}

int main()
{
   if (!glfwInit())
   {
       std::cerr << "Failed to initialize GLFW\n";
       return -1;
   }

   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
   glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
   glfwWindowHint(GLFW_SAMPLES, 4);

   g_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "FAST Massive 2D Conway's Life + PANNING", NULL, NULL);
   if (g_window == NULL)
   {
       std::cerr << "Failed to create GLFW window\n";
       glfwTerminate();
       return -1;
   }

   glfwMakeContextCurrent(g_window);
   glfwSwapInterval(1);

   glfwSetKeyCallback(g_window, keyCallback);
   glfwSetCursorPosCallback(g_window, mouseCallback);
   glfwSetMouseButtonCallback(g_window, mouseButtonCallback);
   glfwSetScrollCallback(g_window, scrollCallback);

   glEnable(GL_DEPTH_TEST);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   
   glClearColor(0.02f, 0.02f, 0.05f, 1.0f);

   // Setup Dear ImGui context
   IMGUI_CHECKVERSION();
   ImGui::CreateContext();
   ImGuiIO& io = ImGui::GetIO(); (void)io;
   io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

   // Setup Dear ImGui style
   ImGui::StyleColorsDark();

   // Setup Platform/Renderer backends
   ImGui_ImplGlfw_InitForOpenGL(g_window, true);
   ImGui_ImplOpenGL3_Init("#version 300 es");

   initShaders();
   uniforms.init(shaderProgram);

   renderBuffers[0].instances.reserve(MAX_INSTANCES);

   // Cube vertices
   float s = VOXEL_SIZE * 0.3f;
   float cubeVertices[] = {
       -s, -s, -s,  s, -s, -s,  s,  s, -s,  s,  s, -s, -s,  s, -s, -s, -s, -s,
       -s, -s,  s,  s, -s,  s,  s,  s,  s,  s,  s,  s, -s,  s,  s, -s, -s,  s,
       -s,  s,  s, -s,  s, -s, -s, -s, -s, -s, -s, -s, -s, -s,  s, -s,  s,  s,
        s,  s,  s,  s,  s, -s,  s, -s, -s,  s, -s, -s,  s, -s,  s,  s,  s,  s,
       -s, -s, -s,  s, -s, -s,  s, -s,  s,  s, -s,  s, -s, -s,  s, -s, -s, -s,
       -s,  s, -s,  s,  s, -s,  s,  s,  s,  s,  s,  s, -s,  s,  s, -s,  s, -s
   };

   glGenVertexArrays(1, &cubeVAO);
   glGenBuffers(1, &cubeVBO);
   glGenBuffers(1, &instanceVBO);
   glBindVertexArray(cubeVAO);
   glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
   glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

   glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
   glBufferData(GL_ARRAY_BUFFER, MAX_INSTANCES * sizeof(InstanceData), nullptr, GL_STREAM_DRAW);
   glEnableVertexAttribArray(1);
   glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, position));
   glVertexAttribDivisor(1, 1);
   glEnableVertexAttribArray(2);
   glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, color));
   glVertexAttribDivisor(2, 1);

   // Boundary for massive 2D plane
   float w = GRID_WIDTH * VOXEL_SIZE * 0.5f;
   float h = GRID_HEIGHT * VOXEL_SIZE * 0.5f;
   float boundaryVertices[] = {
       -w, 0, -h,  w, 0, -h,
        w, 0, -h,  w, 0,  h,
        w, 0,  h, -w, 0,  h,
       -w, 0,  h, -w, 0, -h
   };
   
   glGenVertexArrays(1, &boundaryVAO);
   glGenBuffers(1, &boundaryVBO);
   glBindVertexArray(boundaryVAO);
   glBindBuffer(GL_ARRAY_BUFFER, boundaryVBO);
   glBufferData(GL_ARRAY_BUFFER, sizeof(boundaryVertices), boundaryVertices, GL_STATIC_DRAW);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
   glBindVertexArray(0);

   std::cout << "=== FAST MASSIVE 2D CONWAY'S GAME OF LIFE + PANNING ===" << std::endl;
   std::cout << "Grid size: " << GRID_WIDTH << " x " << GRID_HEIGHT << " = " 
             << (GRID_WIDTH * GRID_HEIGHT / 1000000.0f) << " million cells" << std::endl;
   std::cout << "Max renderable cells: " << MAX_INSTANCES << std::endl;
   std::cout << std::endl;
   std::cout << "=== CONTROLS ===" << std::endl;
   std::cout << "SIMULATION:" << std::endl;
   std::cout << "  Space: Pause/Resume" << std::endl;
   std::cout << "  R: Reset with random pattern" << std::endl;
   std::cout << "  1/2/3/4: Different size/density patterns" << std::endl;
   std::cout << "  +/-: Adjust simulation speed" << std::endl;
   std::cout << "  H: Toggle ImGui overlay" << std::endl;
   std::cout << "  [/]: Decrease/Increase color spread" << std::endl;
   std::cout << "  C: Cycle colors" << std::endl;
   std::cout << std::endl;
   std::cout << "CAMERA ROTATION:" << std::endl;
   std::cout << "  Arrow keys/WASD: Rotate camera" << std::endl;
   std::cout << "  Left mouse drag: Rotate camera" << std::endl;
   std::cout << "  T: Toggle auto-rotate" << std::endl;
   std::cout << std::endl;
   std::cout << "PANNING:" << std::endl;
   std::cout << "  I/J/K/L keys: Pan up/left/down/right" << std::endl;
   std::cout << "  Numpad 8/4/2/6: Pan up/left/down/right" << std::endl;
   std::cout << "  Middle mouse drag: Pan with mouse" << std::endl;
   std::cout << "  Right mouse drag: Pan with mouse" << std::endl;
   std::cout << "  Home/Numpad-5: Reset pan to center" << std::endl;
   std::cout << std::endl;
   std::cout << "ZOOM:" << std::endl;
   std::cout << "  Q/E: Zoom in/out" << std::endl;
   std::cout << "  Mouse scroll: Zoom" << std::endl;
   std::cout << "=========================================" << std::endl;

   initializeMassive2DPattern();

#ifdef __EMSCRIPTEN__
   emscripten_set_main_loop(mainLoopIteration, 0, 1);
#else
   while (!glfwWindowShouldClose(g_window)) {
       mainLoopIteration();
   }
#endif

   // Cleanup ImGui
   ImGui_ImplOpenGL3_Shutdown();
   ImGui_ImplGlfw_Shutdown();
   ImGui::DestroyContext();

   // Cleanup OpenGL resources
   glDeleteVertexArrays(1, &cubeVAO);
   glDeleteBuffers(1, &cubeVBO);
   glDeleteBuffers(1, &instanceVBO);
   glDeleteVertexArrays(1, &boundaryVAO);
   glDeleteBuffers(1, &boundaryVBO);
   glDeleteProgram(shaderProgram);
   glDeleteProgram(boundaryShaderProgram);

   glfwTerminate();
   return 0;
}