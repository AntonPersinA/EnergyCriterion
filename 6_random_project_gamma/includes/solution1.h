#pragma once

#include <iostream>
#include <algorithm>
#include <cmath>
#include <span>
#include <functional>
#include <chrono>

#include <eigen3/Eigen/Dense>
#include <omp.h>

#include <../stats/include/stats.hpp>

using namespace Eigen;

template <typename T>
void print(const T& s);

template <typename T>
void print_iter(const T& s);


