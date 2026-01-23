cmake_minimum_required(VERSION 3.18)
project(CSGRayCast LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Point to local SDL2 binaries in the sibling directory
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(SDL2_ROOT "${PARENT_DIR}/SDL2-2.32.10")

# Debug: verify path if find_package fails
message(STATUS "SDL2_ROOT: ${SDL2_ROOT}")

list(APPEND CMAKE_PREFIX_PATH "${SDL2_ROOT}")
find_package(SDL2 REQUIRED)

file(GLOB SOURCES "*.cpp" "*.cu")
add_executable(CSGRayCast ${SOURCES})

# SDL2::SDL2main is required for Win32 entry point compatibility
if(TARGET SDL2::SDL2main)
    target_link_libraries(CSGRayCast PRIVATE SDL2::SDL2 SDL2::SDL2main)
else()
    target_link_libraries(CSGRayCast PRIVATE SDL2::SDL2)
endif()

# Ensure SDL2.dll is in the same directory as the executable
add_custom_command(TARGET CSGRayCast POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${SDL2_ROOT}/lib/x64/SDL2.dll"
    $<TARGET_FILE_DIR:CSGRayCast>
    COMMENT "Deploying SDL2.dll"
)