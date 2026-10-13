cmake_minimum_required(VERSION 3.28)
project(NIR6_new_2026)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ВАЖНО: Добавляем Release режим по умолчанию
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Оптимизации для конкретного компилятора
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # МИНИМАЛЬНЫЕ ИЗМЕНЕНИЯ: добавляем AVX2 и FMA
    add_compile_options(-O3 -march=native -mtune=native)
    add_compile_options(-ftree-vectorize -ffast-math)
    add_compile_options(-funroll-loops -fomit-frame-pointer)
    add_compile_options(-flto=auto -fno-semantic-interposition)
    add_compile_options(-mavx2 -mfma)

    # ДОБАВЛЕНО: Оптимизация для кэша
    add_compile_options(-falign-loops=64 -falign-functions=64)
    add_compile_options(-fprefetch-loop-arrays)

    add_link_options(-flto=auto -Wl,-O3)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-O3 -march=native -mtune=native -ftree-vectorize -ffast-math)
    add_compile_options(-funroll-loops -flto=thin)
    add_compile_options(-mavx2 -mfma)

    # ДОБАВЛЕНО: Оптимизация для кэша для Clang
    add_compile_options(-falign-loops=64)

    add_link_options(-flto=thin -Wl,-O3)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/O2 /arch:AVX2 /fp:fast /GL)
    add_compile_options(/Qpar /Qpar-report:2)
    add_link_options(/LTCG /OPT:REF /OPT:ICF)
endif()

# OpenMP с проверкой
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(_OPENMP)

    # ДОБАВЛЕНО: оптимизации OpenMP
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS} -fopenmp-simd")
    endif()
endif()

# Boost с оптимизациями
find_package(Boost 1.65 REQUIRED COMPONENTS random)
target_compile_definitions(Boost::random INTERFACE
        BOOST_MATH_DISABLE_FLOAT128
        BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
)

# Собираем библиотеку (статическую для лучшей оптимизации)
add_library(my_lib STATIC
        src/solution1.cpp
)

# Пути к include
target_include_directories(my_lib PUBLIC
        includes
        stats/include
        gcem/include
)

# Предкомпилированные заголовки (ускоряет компиляцию)
target_precompile_headers(my_lib PRIVATE
        <iostream>
        <vector>
        <algorithm>
        <random>
        <chrono>
        <cmath>
)

# Оптимизации для my_lib
target_compile_options(my_lib PRIVATE $<$<CONFIG:RELEASE>:-DNDEBUG>)
target_compile_definitions(my_lib PRIVATE
        NDEBUG
        EIGEN_NO_DEBUG
)

# Связывание
target_link_libraries(my_lib PUBLIC
        OpenMP::OpenMP_CXX
        Boost::random
)

# Исполняемый файл
add_executable(NIR6_new_2026 main.cpp)

# Применяем те же оптимизации к main
target_compile_options(NIR6_new_2026 PRIVATE $<$<CONFIG:RELEASE>:-DNDEBUG>)

# Линковка
target_link_libraries(NIR6_new_2026 PRIVATE my_lib)

# Профайлинг (опционально)
option(ENABLE_PROFILING "Enable profiling" OFF)
if(ENABLE_PROFILING)
    target_compile_options(my_lib PRIVATE -pg)
    target_link_options(my_lib PRIVATE -pg)
    target_compile_options(NIR6_new_2026 PRIVATE -pg)
    target_link_options(NIR6_new_2026 PRIVATE -pg)
endif()

# Статическая линковка (опционально)
option(STATIC_LINK "Statically link libstdc++" OFF)
if(STATIC_LINK AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(NIR6_new_2026 PRIVATE -static-libstdc++ -static-libgcc)
endif()

# Вывод информации
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Compiler: GCC ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "Optimizations: AVX2 + FMA + LTO + Cache")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Compiler: Clang ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "Optimizations: AVX2 + FMA + ThinLTO + Cache")
endif()
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "Boost: ${Boost_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")