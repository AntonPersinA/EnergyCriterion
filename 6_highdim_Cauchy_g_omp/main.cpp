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

#define BOOST_MATH_DISABLE_FLOAT128
#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS

using namespace Eigen;
using namespace std;

const int MONTE_CARLO_CNT = 1e7; // 1e8 - 5 секунд при dim = 1
const int EMPIRICAL_CNT = 1e4; // для подсчета эмпирического среднего и ковариации

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

template<int dim, typename Generator>
class Multivariate_Distribution {
    Generator &gen_sample; // таким образом задаем распределение
    double (*g)(double); // задаем g
    double (*ddg)(double); // задаем ddg

    double J_1[dim]{};
    double J_2[dim]{};
    double J_3[dim]{};
    double J_1_star[dim]{};
    double J_2_star[dim]{};
public:
    Multivariate_Distribution(Generator &gen_sample, double (*g)(double),
                              double (*ddg)(double)) : gen_sample(gen_sample), g(g), ddg(ddg) {

    }

    ~Multivariate_Distribution() = default;

    const double* get_J_1() const { return J_1; }
    const double* get_J_2() const { return J_2; }
    const double* get_J_3() const { return J_3; }
    const double* get_J_1_star() const { return J_1_star; }
    const double* get_J_2_star() const { return J_2_star; }

    void compute_const() {
        double x[dim]{};
        double y[dim]{};
        double z[dim]{};

        double x_minus_y[dim];
        double x_minus_z[dim];
        for (int i = 0; i < MONTE_CARLO_CNT; ++i) { // считаем интегралы методом Монте-Карло
            gen_sample.generate_scale(x);
            gen_sample.generate_scale(y);
            gen_sample.generate_scale(z);
            double inv = 1 / (double)(i+1);

            for (int j = 0; j < dim; ++j) {
                x_minus_y[j] = x[j] - y[j];
                x_minus_z[j] = x[j] - z[j];
            }

            for (int j = 0; j < dim; ++j) {
                J_1[j] = J_1[j]*(i*inv) + inv * g(x_minus_y[j]);
                J_2[j] = J_2[j]*(i*inv) + inv * square( g(x_minus_y[j]) );
                J_3[j] = J_3[j]*(i*inv) + inv * g(x_minus_y[j]) * g(x_minus_z[j]);
                J_1_star[j] = J_1_star[j]*(i*inv) + inv * ddg(x_minus_y[j]);
                J_2_star[j] = J_2_star[j]*(i*inv) + inv * (y[j]*y[j] - 0.5*square(x_minus_y[j])) * ddg(x_minus_y[j]);
            }
        }
        for (int j = 0; j < dim; ++j) {
            J_1_star[j] *= 0.5; // J_1_star при h_1 = 1, если хочется при любой h_1, то нужно просто J_1_star * h_1^2
            J_2_star[j] *= 0.5; // J_2_star при h_2 = 1, если хочется при любой h_2, то нужно просто J_2_star * h_2^2
        }
    }
};


template<int dim>
class CauchyGenerator {
public:
    std::cauchy_distribution<> cauchy{0.0, 1.0};

    std::mt19937_64 &engine;
    double mean[dim]{}; // сдвиг
    double std_dev[dim][dim]{}; // масштаб(стандартное отклонение)

    double empirical_mean[dim]{};
    double empirical_std_dev[dim][dim]{};

    double std_multi_cauchy[dim]{}; // используется для заполнение(ускоряет работу)
    double shift[dim]{}; // задает сдвиг при генерации
    double transformation[dim][dim]{}; // задает насколько масштабировать выборку
public:
    explicit CauchyGenerator(std::mt19937_64 &engine, double (&mean)[dim], double (&variance)[dim][dim]) : engine(engine) {
        auto mtrx_std = matrix_sqrt<dim>(variance, std_dev);
        for (int i = 0; i < dim; ++i) {
            this->mean[i] = mean[i];
        }
        compute_const();
    }

    ~CauchyGenerator() = default;

    void operator()(double x[dim]) {
        for (int i = 0; i < dim; ++i) {
            std_multi_cauchy[i] = cauchy(engine);
        }
#pragma GCC unroll 2
        for (int i = 0; i < dim; ++i) {
            x[i] = mean[i];
#pragma GCC unroll 2
            for (int j = 0; j < dim; ++j) {
                x[i] += std_multi_cauchy[j] * std_dev[i][j];
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
        std::string res = "Cauchy(dim = " + std::to_string(dim) + ")";
        return res;
    }
};


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
class StudentGenerator {
public:
    NormalGenerator<dim> normal{};
    std::gamma_distribution<double> gamma_dist;

    std::mt19937_64& engine;
    double nu; // степени свободы
    double mean[dim]{}; // теоретическое среднее
    double std_dev[dim][dim]{}; // теоретическое стандартное отклонение(корень Cov)

    double empirical_mean[dim]{}; // существует при nu > 1
    double empirical_std_dev[dim][dim]{}; // существует при nu > 2

    double std_multi_normal[dim]{}; // используется для заполнения нормальными величинами(ускоряет работу)
    double shift[dim]{}; // задает сдвиг при генерации
    double transformation[dim][dim]{}; // задает насколько масштабировать выборку

public:
    explicit StudentGenerator(std::mt19937_64& engine,
                              double (&location)[dim], double (&scale_matrix)[dim][dim], double nu = 20)
            : engine(engine), nu(nu), normal(engine, location, scale_matrix), gamma_dist(nu / 2.0, 2.0) {
        for (int i = 0; i < dim; ++i) {
            mean[i] = location[i];
            for (int j = 0; j < dim; ++j) {
                std_dev[i][j] = normal.std_dev[i][j] * sqrt(nu / (nu-2));
            }
        }
        compute_const();
    }

    ~StudentGenerator() = default;

    void operator()(double x[dim]) {
        normal(std_multi_normal);
        double w = gamma_dist(engine);
#pragma GCC unroll 2
        for (int i = 0; i < dim; ++i) {
            x[i] = mean[i] + (std_multi_normal[i] - mean[i]) * sqrt(nu / w);
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
        std::string res = "Student(dim = " + std::to_string(dim) + ", nu = " + std::to_string(nu) + ")";
        return res;
    }
};


inline double g(double x) {
    return x*x;
}

inline double ddg(double x) {
    return 2;
}


//inline double g(double x) {
//    return abs(x);
//}
//
//inline double ddg(double x) {
//    return 0;
//}


//inline double g(double x) {
//    return log1p(x*x);
//}
//
//inline double ddg(double x) {
//    return - 2 * (x*x - 1) / square(1 + x*x);
//}


//inline double g(double x) {
//    return log1p(x*x) + x*x;
//}
//
//inline double ddg(double x) {
//    return - 2 * (x*x - 1) / square(1 + x*x) + 2;
//}


template<int dim>
double T(double X[][dim], double Y[][dim], size_t n, double (*g)(double)) // X - выборка размера n из распределения F_1, Y - выборка размера n из распределения F_2
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
            res += g(sqrt(normXiYj)) - 0.5 * ( g(sqrt(normXiXj)) + g(sqrt(normYiYj)) );
        }
    }
    return res / (double)(n*n);
}


int main()
{
//    omp_set_num_threads(1); // параллелизация = закомментировать
    std::ofstream fout("results.csv");

    std::normal_distribution<double> normal(0.0, 1.0);
    std::mt19937_64 engine(11);
    const int dim = 2;
    double mean[dim] = {0, 0};
    double variance[dim][dim] = {{1.0, 0.0},
                                 {0.0, 1.0}}; // по какой-то причине при коррелируемости теор мощность возрастает, хотя должна быть той же. Так что лучше некоррелированно запускать

    // Задаем первое распределение
    using DistTamplate = CauchyGenerator<dim>; // здесь дисперсия(параметр inv_sigma_matrix ниже) должен быть вычислен не через оценку, а "точно", так как дисперсии нет
//    using DistTamplate = StudentGenerator<dim>; // здесь дисперсия должна быть "точной", если nu <= 2, мат ожидание должно быть точным, если nu <= 1
//    using DistTamplate = NormalGenerator<dim>;
    DistTamplate dist_0(engine, mean, variance);

    // Задаем гиперпараметры
    const std::vector<size_t> N = {100, 400, 900, 1500, 3000}; // 100, 400, 900
    const std::vector<double> H_1 = {0, 0.5, 2, 7}; // 0, 0.5, 1, 2, 3, 4, 5, 6, 7
    double h_2 = 0;
    double alpha = 0.05;

    // TheoreticalPower params
    int sample_size_q = 1e6; // определяет, с какой точностью искать alpha-квантиль. 1e8 = 16Гб памяти, так что лучше брать меньше 1e7 = 160Mb
    int sample_size_P = sample_size_q; // определяет, с какой точностью ищется сама мощность.

    // EmpiricalPower params
    size_t M = 1000; // 5000 - размер выборки, из которой берется квантиль статистики(больше ничего с этой переменной не происходит!!?)
    size_t exp_cnt = 10; // 10 - считаем exp_cnt раз мощность и усредняем
    size_t num_simulations = 1000; // 2000 - определяет точность, с которой ищется мощность


    // Вывод параметров
    {
        std::cout << dist_0.get_name() << std::endl;
        std::cout << "mean = (" << mean[0];
        for (int i = 1; i < dim; ++i) {
            std::cout << ", " << mean[i];
        }
        std::cout << ")" << std::endl;

        std::cout << "var = [";
        for (int i = 0; i < dim; ++i) {
            if (i > 0) std::cout << "       [";
            for (int j = 0; j < dim; ++j) {
                std::cout << variance[i][j];
                if (j < dim-1) std::cout << ", ";
            }
            std::cout << "]";
            if (i < dim-1) std::cout << std::endl;
        }
        std::cout << std::endl;
        printf("h_2 = %1.2f, alpha = %1.2f\n", h_2, alpha);
        printf("M = %ld, exp_cnt = %ld, num_simulations = %ld\n", M, exp_cnt, num_simulations);
        printf("MONTE_CARLO_CNT = %1.e, EMPIRICAL_CNT = %1.e, sample_size_q = %1.e, sample_size_P = %1.e\n\n",
               (double)MONTE_CARLO_CNT, (double)EMPIRICAL_CNT, (double)sample_size_q, (double)sample_size_P);
    }

    // Запись параметров в файл
    {
        fout << dist_0.get_name() << std::endl;
        for (int i = 0; i < dim; ++i) {
            fout << mean[0] << " ";
        }
        fout << std::endl;
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                fout << variance[i][j] << " ";
            }
        }
        fout << std::endl << h_2 << " " << alpha << std::endl;
        fout << "type,n,h1,value" << std::endl;
    }

    // Считаем коэффициенты масштабирования
    double minus_E_x[dim]{};
    for (int i = 0; i < dim; ++i) {
        minus_E_x[i] = -dist_0.empirical_mean[i];
    }
    auto inv_sigma_matrix = inverseMatrixLDLT<dim>(dist_0.empirical_std_dev); // делает признаки некоррелированными "эмпирически", то есть ковариация считается
//    auto inv_sigma_matrix = inverseMatrixLDLT<dim>(dist_0.std_dev); // делает признаки некоррелированными "теоретически"
    double inv_sigma_x[dim][dim]{};
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
//            inv_sigma_x[i][j] = inv_sigma_matrix(i, j);
            inv_sigma_x[i][j] = inv_sigma_matrix(i, j) / inv_sigma_matrix.maxCoeff(); // делим на максимум, чтобы(например Коши) inv_sigma_x(обратная к корню дисперсии) было не нулевым
        }
    }
    dist_0.set_scale(minus_E_x, inv_sigma_x);


// считаем интегралы J_1, ...,J_2_star методом Монте-Карло
    Multivariate_Distribution<dim, DistTamplate> CC(dist_0, g, ddg); // ComputeConst
    CC.compute_const();

    std::string A_printf = "";
// подсчет TheoreticalPower
    double TheoreticalPower = 0;
    auto *H_0_lim_dist_sample = new double[sample_size_q];

    for (double h_1 : H_1) {
        double b_1[dim]{}, b_2[dim]{}, b[dim]{}, a_square[dim]{}, a[dim]{};
        for (int i = 0; i < dim; ++i) {
            b_1[i] = sqrt(abs(CC.get_J_1_star()[i] * h_1*h_1));
            b_2[i] = sqrt(abs(CC.get_J_2_star()[i] * h_2*h_2));
            b[i] = sqrt(b_1[i]*b_1[i] + b_2[i]*b_2[i]);
            a_square[i] = sqrt(CC.get_J_2()[i] + CC.get_J_1()[i]*CC.get_J_1()[i] - 2*CC.get_J_3()[i]);
            a[i] = sqrt(a_square[i]);
        }

        for (int i = 0; i < sample_size_q; ++i) {
            H_0_lim_dist_sample[i] = 0;
            for (int j = 0; j < dim; ++j) {
                H_0_lim_dist_sample[i] += a_square[j] * square(normal(engine));
            }
        }
        std::sort(H_0_lim_dist_sample, H_0_lim_dist_sample + sample_size_q);
        auto q_index = static_cast<size_t>(std::floor((1 - alpha) * (sample_size_q - 1))); // эмпирическая 1 - alpha квантиль
        double z = H_0_lim_dist_sample[q_index];

        int cnt = 0;
        for (int i = 0; i < sample_size_P; ++i) { // здесь для одномерного случая. для многомерного смотри формулу номер (9) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            double stats = 0;
            for (int j = 0; j < dim; ++j) {
                stats += square( a[j] * normal(engine) + b[j]);
            }
            if (stats > z) {
                ++cnt;
            }
        }
        TheoreticalPower = cnt / static_cast<double>(sample_size_P);
        std::cout << "(A) h_1=" << h_1 << " → " << TheoreticalPower << "\n";

        std::ostringstream oss;
        oss << "A,0," << h_1 << "," << std::fixed << std::setprecision(3) << TheoreticalPower << "\n";
        A_printf += oss.str();
    }
    delete[] H_0_lim_dist_sample;


// подсчет EmpiricalPower
    for (size_t n : N) {
        for (double h_1 : H_1) {
            double m_1[dim]{};
            double variance_1[dim][dim]{};
            for (int i = 0; i < dim; ++i) {
                m_1[i] = mean[i] + h_1/sqrt(n);
                for (int j = 0; j < dim; ++j) {
                    variance_1[i][j] = variance[i][j] * (1 + h_2/sqrt(n));
                }
            }
            double EmpiricalPower = 0;
            auto start = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for reduction(+:EmpiricalPower)
            for (int exp = 0; exp < exp_cnt; ++exp) {
                double XY[2*n][dim];
                double STATS[M];
                int tid = omp_get_thread_num();
                std::mt19937_64 local_engine_0(11 + exp * 100000 + tid * 10000 + 0);  // для dist_0
                std::mt19937_64 local_engine_1(11 + exp * 100000 + tid * 10000 + 1);  // для dist_1
                std::mt19937_64 local_engine_shuffle(11 + exp * 100000 + tid * 10000 + 2); // для shuffle

                DistTamplate local_dist_0(local_engine_0, mean, variance);
                local_dist_0.set_scale(minus_E_x, inv_sigma_x);
                DistTamplate local_dist_1(local_engine_1, m_1, variance_1);
                local_dist_1.set_scale(minus_E_x, inv_sigma_x);

                for (int i = 0; i < 2 * n; ++i) { // заполняем X, Y
                    local_dist_0.generate_scale(XY[i]);
                }

                for (int i = 0; i < M; ++i) {
                    std::shuffle(XY, XY + 2*n, local_engine_shuffle);
                    STATS[i] = T<dim>(XY, XY + n, n, g);
                }
                std::sort(STATS, STATS + M);
                auto q_index = static_cast<size_t>(std::floor((1 - alpha) * ((int)M - 1))); // эмпирическая 1 - alpha квантиль
                double critical_value = STATS[q_index];

                int rejections = 0;
                for (int sim = 0; sim < num_simulations; ++sim) {

                    for (int i = 0; i < n; ++i) {
                        local_dist_0.generate_scale(XY[i]);
                    }
                    for (int i = (int)n; i < 2 * n; ++i) {
                        local_dist_1.generate_scale(XY[i]);
                    }
//                    if (sim == 1 and T<dim>(XY, XY + n, n, g) > critical_value) {
//                        std::cout << "T<dim>(XY, XY + n, n, g)" << T<dim>(XY, XY + n, n, g) << "crit_val = " << critical_value << std::endl;
//                    }
                    if (T<dim>(XY, XY + n, n, g) > critical_value) {
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


            fout << "E," << n << "," << h_1 << "," << std::fixed << std::setprecision(3) << EmpiricalPower << std::endl;
        }
    }
    fout << A_printf;
    fout.close();
    return 0;
}