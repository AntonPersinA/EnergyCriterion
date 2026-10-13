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
const int EMPIRICAL_CNT = 1e3; // для подсчета эмпирического среднего и ковариации

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


double Energy_Statistic(double X[], double Y[], size_t n, size_t m)
{
    sort(X, X + n);
    sort(Y, Y + m);

    struct Item {
        double value;
        int fromX; // 1 если из X, 0 если из Y
    };
    Item Z[n + m];
    for (int i = 0; i < n; ++i) {
        Z[i] = {X[i], 1};
    }
    for (int i = 0; i < m; ++i) {
        Z[n + i] = {Y[i], 0};
    }
    sort(Z, Z + n + m, [](const Item& a, const Item& b) {
        return a.value < b.value;
    });

    double E_1 = 0;
    int I_i = 0; // сколько элементов из X встретилось
    for (int i = 1; i < n + m; ++i) {
        if (Z[i-1].fromX) I_i++; // из X среди Z[0..i] = countX, из Y среди Z[0..i] = (i+1) - countX
        E_1 += (I_i*(m - i + I_i) + (i - I_i)*(n - I_i)) * abs(Z[i].value - Z[i-1].value);
    }
    E_1 *= 2.0 / (n*m);


    double E_2 = 0;
    for (int i = 1; i < n; ++i) {
        E_2 += i*(n - i) * abs(X[i] - X[i-1]);
    }
    E_2 *= 2.0 / (n*(n - 1));


    double E_3 = 0;
    for (int i = 1; i < m; ++i) {
        E_3 += i*(m - i) * abs(Y[i] - Y[i-1]);
    }
    E_3 *= 2.0 / (m*(m - 1));
    return E_1 - E_2 - E_3;
}


template<int dim>
double scalar(double u[dim], double v[dim])
{
    double res = 0;
    for (int i = 0; i < dim; ++i) {
        res += u[i]*v[i];
    }
    return res;
}

double c_p(int dim)
{
    if (dim == 0) return 1;
    return pow(M_PI, (dim + 1)/2.) / std::tgamma((dim+1)/2.);
}

double C_p(int dim)
{
    if (dim <= 1) return 1;
    double res = pow(M_PI, 0.5);
    double i = dim/2.0;
    for (; i > 1; --i) {
        res *= (i + 0.5 - 1) / (i - 1);
    }
    res *= std::tgamma(i + 0.5) / std::tgamma(i);
    return res;
}

template<int dim>
double RPES(double X[][dim], double Y[][dim], size_t n, size_t m, int K, std::mt19937_64 &engine)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    const double C_dim = C_p(dim);
    double res = 0;

    double u[dim];
    for (int k = 0; k < K; ++k) {
        double u_norm = 0;
        for (int i = 0; i < dim; ++i) {
            u[i] = normal(engine);
            u_norm += square(u[i]);
        }
        u_norm = sqrt(u_norm);
        for (int i = 0; i < dim; ++i) {
            u[i] /= u_norm;
        }


        double uX[n];
        double uY[m];
        for (int i = 0; i < n; ++i) {
            uX[i] = scalar<dim>(u, X[i]);
        }
        for (int i = 0; i < m; ++i) {
            uY[i] = scalar<dim>(u, Y[i]);
        }

        res += C_dim * Energy_Statistic(uX, uY, n, m);
    }
    return res / K;
}


template<int dim>
double RPES_test(double X[][dim], double Y[][dim], size_t n, size_t m, int K, int L, double alpha, std::mt19937_64 &engine)
{
    double crit_value = RPES(X, Y, n, m, K, engine);
    double Z[n + m][dim];
    double X_perm[n][dim];
    double Y_perm[m][dim];

    for (size_t i = 0; i < n; ++i) // X в начало
        for (int d = 0; d < dim; ++d)
            Z[i][d] = X[i][d];
    for (size_t i = 0; i < m; ++i) // Y в конец
        for (int d = 0; d < dim; ++d)
            Z[n + i][d] = Y[i][d];

    int rejections = 0;
    for (int l = 0; l < L; ++l) {
        std::shuffle(Z, Z + n + m, engine); // Перемешиваем Z

        for (size_t i = 0; i < n; ++i)
            for (int d = 0; d < dim; ++d)
                X_perm[i][d] = Z[i][d];  // новые X
        for (size_t i = 0; i < m; ++i)
            for (int d = 0; d < dim; ++d)
                Y_perm[i][d] = Z[n + i][d]; // новые Y


        auto t = RPES(X_perm, Y_perm, n, m, K, engine);
//        std::cout << "t = " << t << ", crit_val = " << crit_value << std::endl;
        if (crit_value < t) { // ?????????????????????????????????????????????????????????????????????????????????????????? в статье crit_value > t
            ++rejections;
        }
    }
//    std::cout << "p-val = " << ((1 + rejections) / (double)(1 + L)) << std::endl;
    return ((1 + rejections) / (double)(1 + L)) < alpha; // 1 == H_0 отвергаем, 0 не отвергаем(верна)
}


int main()
{
    std::ofstream fout("results.csv");

    std::normal_distribution<double> normal(0.0, 1.0);
    std::mt19937_64 engine(11);
    const int dim = 2;
    double mean[dim] = {0, 0};
    double variance[dim][dim] = {{1.0, 0.0},
                                 {0.0, 1.0}}; // по какой-то причине при коррелируемости теор мощность возрастает, хотя должна быть той же. Так что лучше некоррелированно запускать

    // Задаем первое распределение
//    using DistTamplate = CauchyGenerator<dim>; // здесь дисперсия(параметр inv_sigma_matrix ниже) должен быть вычислен не через оценку, а "точно", так как дисперсии нет
//    using DistTamplate = StudentGenerator<dim>; // здесь дисперсия должна быть "точной", если nu <= 2, мат ожидание должно быть точным, если nu <= 1
    using DistTamplate = NormalGenerator<dim>;
    DistTamplate dist_0(engine, mean, variance);

    // Задаем гиперпараметры
    const std::vector<size_t> N = {100, 400, 900}; // 100, 400, 900
    const std::vector<double> H_1 = {0, 0.5, 1, 2, 3, 4, 5, 6, 7}; // 0, 0.5, 1, 2, 3, 4, 5, 6, 7
    double h_2 = 0;
    double alpha = 0.05;

    // TheoreticalPower params
    int sample_size_q = 1e6; // определяет, с какой точностью искать alpha-квантиль. 1e8 = 16Гб памяти, так что лучше брать меньше 1e7 = 160Mb
    int sample_size_P = sample_size_q; // определяет, с какой точностью ищется сама мощность.

    // EmpiricalPower params
    size_t K = 50;
    size_t L = 1000; // 5000 - размер выборки, из которой берется квантиль статистики(больше ничего с этой переменной не происходит!!?)
    size_t num_simulations = 200; // 2000 - определяет точность, с которой ищется мощность


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


//    std::mt19937_64 local_engine_0(11);  // для dist_0
//    std::mt19937_64 local_engine_1(13);  // для dist_1
//    std::mt19937_64 local_engine_shuffle(17); // для shuffle

    for (size_t n : N) {
        size_t m = n;
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

            int rejections = 0;
            #pragma omp parallel for reduction(+:rejections)
            for (int sim = 0; sim < num_simulations; ++sim) {
                double X[n][dim];
                double Y[m][dim];

                int tid = omp_get_thread_num();
                std::mt19937_64 local_engine_0(sim * 100000 + tid * 10000 + 11);  // для dist_0
                std::mt19937_64 local_engine_1(sim * 100000 + tid * 10000 + 13);  // для dist_1
                std::mt19937_64 local_engine_shuffle(sim * 100000 + tid * 10000 + 17); // для shuffle

                DistTamplate local_dist_0(local_engine_0, mean, variance);
                DistTamplate local_dist_1(local_engine_1, m_1, variance_1);

                for (int i = 0; i < n; ++i) { // заполняем X
                    local_dist_0(X[i]);
                }
                for (int i = 0; i < m; ++i) { // заполняем Y
                    local_dist_1(Y[i]);
                }

                auto test = RPES_test(X, Y, n, m, K, L, alpha, local_engine_shuffle);

                rejections += test;
//                std::cout << "sim = " << sim << ", reject/all_sim = " << rejections << "/" << sim + 1 << " = " << (double) rejections / (double) (sim + 1) << std::endl;
            } // end of for(int exp = 0; exp < exp_cnt; ++exp)
            EmpiricalPower += (double)rejections / (double)num_simulations;

            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end - start);

            std::cout << "n = " << n << ", h_1 = " << h_1 << ", EP = " << EmpiricalPower << ", Время 2: " << duration_sec.count() << " сек "
                      << duration_ms.count() % 1000 << " мс" << std::endl;


            fout << "E," << n << "," << h_1 << "," << std::fixed << std::setprecision(3) << EmpiricalPower << std::endl;
        }
    }
    fout.close();


    return 0;
}