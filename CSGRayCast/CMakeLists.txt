cmake_minimum_required(VERSION 3.18)
project(CSGRayCast LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(SDL2 CONFIG QUIET)

if(NOT TARGET SDL2::SDL2)
    include(FetchContent)

    set(SDL_SHARED ON CACHE BOOL "Build a shared SDL library" FORCE)
    set(SDL_STATIC OFF CACHE BOOL "Build a static SDL library" FORCE)
    set(SDL_TEST OFF CACHE BOOL "Build SDL tests" FORCE)

    FetchContent_Declare(
        SDL2
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG release-2.32.10
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(SDL2)
endif()

file(GLOB SOURCES "*.cpp" "*.cu")
add_executable(CSGRayCast ${SOURCES})

# SDL2::SDL2main is required for Win32 entry point compatibility
if(TARGET SDL2::SDL2main)
    target_link_libraries(CSGRayCast PRIVATE SDL2::SDL2 SDL2::SDL2main)
else()
    target_link_libraries(CSGRayCast PRIVATE SDL2::SDL2)
endif()

# Deploy runtime dependencies next to the executable on Windows.
if(WIN32)
    add_custom_command(TARGET CSGRayCast POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:CSGRayCast>
        $<TARGET_FILE_DIR:CSGRayCast>
        COMMAND_EXPAND_LISTS
        COMMENT "Deploying runtime dependencies"
    )
endif()
