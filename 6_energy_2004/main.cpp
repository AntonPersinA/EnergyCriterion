#include <iostream>
#include <chrono>
#include <fstream>
#include <boost/random.hpp>
#include <eigen3/Eigen/Dense>
#include <../stats/include/stats.hpp>
#include <cmath>
#include <omp.h>
#include <iomanip>

#include "boost/random/cauchy_distribution.hpp"
#include <boost/math/distributions/gamma.hpp>
//#include "boost/random/normal_distribution.hpp"
//#include "boost/math/distributions.hpp"

using namespace Eigen;
using namespace std;

const int EMPIRICAL_CNT = 1e6; // для подсчета эмпирического среднего и ковариации




inline double square(double x) {
    return x*x;
}


template<int dim>
MatrixXd matrix_sqrt(double A[dim][dim], double res[dim][dim])
{
    MatrixXd M(dim, dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            M(i, j) = A[i][j];
        }
    }

    // Спектральное разложение и извлечение корня
    SelfAdjointEigenSolver<MatrixXd> solver(M);
    MatrixXd sqrt_M = solver.eigenvectors() *
                      solver.eigenvalues().cwiseSqrt().asDiagonal() *
                      solver.eigenvectors().transpose();

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            res[i][j] = sqrt_M(i, j);
        }
    }
    return sqrt_M;
}


template<int dim>
Eigen::MatrixXd inverseMatrixLDLT(double A[dim][dim])
{
    Eigen::MatrixXd matrix_A(dim, dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            matrix_A(i, j) = A[i][j];
        }
    }

    Eigen::LDLT<Eigen::MatrixXd> ldlt(matrix_A);
    if (ldlt.info() != Eigen::Success) {
        throw std::invalid_argument("Матрица не является положительно определенной");
    }

    return ldlt.solve(Eigen::MatrixXd::Identity(matrix_A.rows(), matrix_A.cols()));
}


template<int dim>
class NormalGenerator {
public:
    std::normal_distribution<double> std_normal{0.0, 1.0};

    std::mt19937_64 &engine;
    double mean[dim]{}; // сдвиг
    double std_dev[dim][dim]{}; // масштаб(стандартное отклонение)

    double empirical_mean[dim]{};
    double empirical_std_dev[dim][dim]{};

    double std_multi_norm[dim]{}; // используется для заполнение(ускоряет работу)
    double shift[dim]{}; // задает сдвиг при генерации
    double transformation[dim][dim]{}; // задает насколько масштабировать выборку
public:
    explicit NormalGenerator(std::mt19937_64 &engine, double (&mean)[dim], double (&variance)[dim][dim]) : engine(engine) {
        auto mtrx_std = matrix_sqrt<dim>(variance, std_dev);
        for (int i = 0; i < dim; ++i) {
            this->mean[i] = mean[i];
        }
        compute_const();
    }

    ~NormalGenerator() = default;

    void operator()(double x[dim]) {
        for (int i = 0; i < dim; ++i) {
            std_multi_norm[i] = std_normal(engine);
        }
#pragma GCC unroll 2
        for (int i = 0; i < dim; ++i) {
            x[i] = mean[i];
#pragma GCC unroll 2
            for (int j = 0; j < dim; ++j) {
                x[i] += std_multi_norm[j] * std_dev[i][j];
            }
        }
    }

    void compute_empirical_mean() {
        for (int i = 0; i < dim; ++i) {
            empirical_mean[i] = 0;
        }
        double x[dim];
        for (int i = 0; i < EMPIRICAL_CNT; ++i) {
            double inv = 1 / (double)(i+1);
            (*this)(x);
            for (int j = 0; j < dim; ++j) {
                empirical_mean[j] = empirical_mean[j]*(i*inv) + inv * x[j];
            }
        }
    }

    void compute_const() { // считает и empirical_mean, и empirical_std_dev
        compute_empirical_mean();
        double x[dim];
        double empirical_D[dim][dim]{};
        for (int i = 0; i < EMPIRICAL_CNT; ++i) { // считаем дисперсию
            double inv = 1 / (double)(i+1);
            (*this)(x);
            for (int j = 0; j < dim; ++j) {
                for (int k = j; k < dim; ++k) {
                    empirical_D[j][k] = empirical_D[j][k]*(i*inv) + inv * ( x[j]-empirical_mean[j] ) * ( x[k]-empirical_mean[k] );
                }
            }
        }
        for (int j = 0; j < dim; ++j) { // симметрично дописываем вторую половину ковариационной матрицы
            for (int k = 0; k < j; ++k) {
                empirical_D[j][k] = empirical_D[k][j];
            }
        }

        matrix_sqrt<dim>(empirical_D, empirical_std_dev); // извлекаем корень из дисперсии
    }


    void set_scale(const double m[dim], const double A[dim][dim]) {
        for (int i = 0; i < dim; ++i) {
            shift[i] = m[i];
            for (int j = 0; j < dim; ++j) {
                transformation[i][j] = A[i][j];
            }
        }
    }

    void generate_scale(double x[dim]) {
        double time_var[dim];
        (*this)(time_var);
#pragma GCC unroll 2
        for (int i = 0; i < dim; ++i) {
            x[i] = shift[i];
#pragma GCC unroll 2
            for (int j = 0; j < dim; ++j) {
                x[i] += transformation[i][j] * time_var[j];
            }
        }
    }

    std::string get_name() {
        std::string res = "Normal(dim = " + std::to_string(dim) + ")";
        return res;
    }
};

template<int dim>
double Eps(double X[][dim], double Y[][dim], size_t n) // X - выборка размера n из распределения F_1, Y - выборка размера n из распределения F_2
{
    double res = 0;
#pragma omp simd collapse(2) reduction(+:res)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) { // тут можно сократить вычисления вдвое, если считать j < n, но тогда нужно, чтобы g(0) = 0 И ТОГДА РЕЗУЛЬТАТ НУЖНО БУДЕТ НА 2 ДОМНОЖИТЬ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            double normXiXj = 0;
            double normYiYj = 0;
            double normXiYj = 0;
            for (int k = 0; k < dim; ++k) {
                normXiXj += square(X[i][k] - X[j][k]);
                normYiYj += square(Y[i][k] - Y[j][k]);
                normXiYj += square(X[i][k] - Y[j][k]);
            }
            res += 2*sqrt(normXiYj) - sqrt(normXiXj) - sqrt(normYiYj);
        }
    }
    return res / (double)(2*n);
}

std::mt19937_64 engine_shuffle(11);

template<int dim>
double energy_test_2004(double X[][dim], double Y[][dim], size_t n, size_t M = 1000) // return p_value
{
    double XY[2*n][dim];
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            XY[i][d] = X[i][d];
            XY[i+n][d] = Y[i][d];
        }
    }

    double t_critical = Eps<dim>(XY, XY + n, n);

    int rejected = 0;
    for (int i = 0; i < M; ++i) {
        std::shuffle(XY, XY + 2*n, engine_shuffle);
        if (Eps<dim>(XY, XY + n, n) > t_critical) {
            ++rejected;
        }
    }
    return (double)(rejected+1) / (double)(M+1);
}

int main()
{
    std::ofstream fout("results.csv");

    std::normal_distribution<double> normal(0.0, 1.0);
    std::mt19937_64 engine(11);
    const int dim = 2;
    double mean[dim] = {0, 0};
    double variance[dim][dim] = {{1.0, 0.0},
                                 {0.0, 1.0}};

    // Задаем первое распределение
    using DistTamplate = NormalGenerator<dim>;
    DistTamplate dist_0(engine, mean, variance);

    // Задаем гиперпараметры
    const std::vector<size_t> N = {15}; // 100, 400, 900
    const std::vector<double> H_1 = {0, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double h_2 = 0;
    double alpha = 0.05;

    // EmpiricalPower params
    size_t M = 1000; // 5000 - с какой точностью считать p-value
    size_t exp_cnt = 10; // 10 - считаем exp_cnt раз мощность и усредняем
    size_t num_simulations = 1000; // 2000 - определяет точность, с которой ищется мощность

    for (size_t n : N) {
        for (double h_1: H_1) {
            double m_1[dim]{};
            double variance_1[dim][dim]{};
            for (int i = 0; i < dim; ++i) {
                m_1[i] = mean[i] + h_1/sqrt(n);
                for (int j = 0; j < dim; ++j) {
                    variance_1[i][j] = variance[i][j] * (1 + h_2/sqrt(n));
                }
            }
            DistTamplate dist_1(engine, m_1, variance_1);

            double EmpiricalPower = 0;
            auto start = std::chrono::high_resolution_clock::now();

            for (int exp = 0; exp < exp_cnt; ++exp) {
                double X[n][dim];
                double Y[n][dim];

                int rejections = 0;
                for (int sim = 0; sim < num_simulations; ++sim) {
                    for (int i = 0; i < n; ++i) { // заполняем X, Y
                        dist_0(X[i]);
                        dist_1(Y[i]);
                    }

                    if (alpha > energy_test_2004<dim>(X, Y, n, M)) {
                        ++rejections;
                    }
                }
                EmpiricalPower += (double)rejections / (double)num_simulations;
            } // end of for(int exp = 0; exp < exp_cnt; ++exp)
            EmpiricalPower /= (double)exp_cnt;

            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end - start);

            std::cout << "n = " << n << ", h_1 = " << h_1 << ", EP = " << EmpiricalPower << ", Время 2: " << duration_sec.count() << " сек "
                      << duration_ms.count() % 1000 << " мс" << std::endl;

        }
    }

    return 0;
}