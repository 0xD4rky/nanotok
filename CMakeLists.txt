cmake_minimum_required(VERSION 3.15)
project(nanotok LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

pybind11_add_module(_nanotok_cpp csrc/bindings.cpp)

target_include_directories(_nanotok_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_link_libraries(_nanotok_cpp PRIVATE nlohmann_json::nlohmann_json)

install(TARGETS _nanotok_cpp LIBRARY DESTINATION nanotok)
