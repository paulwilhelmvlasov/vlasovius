//============================================================================
// Name        : Particle-In-Cell.cpp
// Author      : Rostislav-Paul Wilhelm
// Version     :
// Copyright   : This is a custom PIC code to solve Vlasov-Poisson. It goes under a GNU-GPL.
// Description : Hello World in C++, Ansi-style
//============================================================================


#include <cblas.h>
#include <iostream>
#include <armadillo>

#include <vlasovius/misc/stopwatch.h>

inline double f0_1d(double x, double v)
{
    using std::cos;
    using std::exp;
    constexpr double alpha = 0.01;
    constexpr double k     = 0.5;
    constexpr double fac   = 0.39894228040143267793994;

    return fac*(1+alpha*cos(k*x))*exp(-v*v/2);
}

inline double shape_function_1d(double x, double eps = 1)
{
	x = x/eps;
	// 2nd order standard univariate B-Spline:
	if(x < 0)
	{
		x = -x;
	}

	if(x > 1)
	{
		return 0;
	}

	return (1 - x)/eps;
}

void init_poiss_matrix(arma::mat &A, double delta_x)
{
	size_t N = A.n_rows;

	// Left border.
    A(0, 0) =  2;
    A(0, 1) = -1;
    // Right border.
    A(N-1, N-1) =  2;
    A(N-1, N-2) = -1;
    // Inside domain.
    for(size_t j = 1; j < N - 1; j++)
    {
    	A(j, j)   =  2;
    	A(j, j-1) = -1;
    	A(j, j+1) = -1;
    }

    A /= delta_x * delta_x;
}

double eval_E(double x, const arma::vec &E, double delta_x)
{
	double E_loc = 0;
	for(size_t i = 0; i < E.n_elem; i++)
	{
		double xi = i*delta_x;
		E_loc += E(i) * shape_function_1d(x - xi, delta_x);
	}

	return E_loc;
}

int main() {
	// This is an implementation of Wang et.al. 2nd-order PIC (see section 3.2).
	// Set parameters.
    const double L  = 4*3.14159265358979323846;
    const size_t Nx_f = 128;
    const size_t Nx_poisson = 16;
    const size_t Nv_f = 512;
    const size_t N_f = Nx_f*Nv_f;
    const double v_min = -10;
    const double v_max =  10;

    // Compute derived quantities.
    const double eps_x = L/Nx_f;
    const double eps_v = (v_max-v_min)/Nv_f;
    const double delta_x = L/Nx_poisson;
    const double delta_x_inv = 1/delta_x;
    const double L_inv  = 1/L;

    const size_t Nt = 100 * 8;
    const double T = 100;
    const double dt = T / Nt;

    // Initiate particles.
    arma::mat xv;
    arma::vec Q(N_f);
    xv.set_size(N_f,2);

    // Initialise xv.
    for ( size_t i = 0; i < Nx_f; ++i )
    for ( size_t j = 0; j < Nv_f; ++j )
    {
        double x =         (i+0.5)*eps_x;
        double v = v_min + (j+0.5)*eps_v;
        size_t k = j+Nv_f*i;
        xv(k, 0 ) = x;
        xv(k, 1 ) = v;
         Q(k)     = eps_x*eps_v*f0_1d(x,v);
    }

    arma::vec rho(Nx_poisson + 1);
    arma::vec varphi(Nx_poisson + 1);
    arma::vec E(Nx_poisson + 1);

    arma::mat poiss_solve_matrix(Nx_poisson + 1, Nx_poisson + 1, arma::fill::zeros);
    init_poiss_matrix(poiss_solve_matrix, delta_x);

    //std::cout << poiss_solve_matrix << std::endl;

    // Time-loop using symplectic Euler.
    std::ofstream str("E.txt");
    double t_total = 0;
    for(size_t nt = 0; nt <= Nt; nt++)
    {
        vlasovius::misc::stopwatch clock;

    	// Compute electron density.
		#pragma omp parallel for
        for(size_t i = 1; i < Nx_poisson; i++)
    	{
    		double x = i*delta_x;
    		rho(i) = 1;
    		for(size_t k = 0; k < N_f; k++)
    		{
    			rho(i) -= Q(k) * shape_function_1d( x - xv(k,0), delta_x);
    		}
    	}
        rho(0) = 0;
        rho(Nx_poisson) = rho(0);

    	// Solve for electric potential.
    	varphi = arma::solve(poiss_solve_matrix, rho);

    	// Compute electric field.
		#pragma omp parallel for
    	for(size_t j = 1; j < Nx_poisson; j++)
    	{
    		E(j) = -0.5*delta_x_inv * (varphi(j+1) - varphi(j-1));
    	}
    	E(0) = 0;
    	E(Nx_poisson) = E(0);

    	// Move in particles.
		#pragma omp parallel for
    	for(size_t k = 0; k < N_f; k++ )
    	{
    		xv(k, 1) -= dt * eval_E(xv(k,0), E, delta_x);
    		xv(k, 0) += dt * xv(k,1);
    		xv(k, 0) -= L*std::floor(xv(k, 0)*L_inv);
    	}

    	double t = nt * dt;
    	double elapsed = clock.elapsed();
    	t_total += elapsed;
    	std::cout << t << "  " << elapsed << " ";

    	// Do analytics.
    	double E_max = E.max();
    	str << t << " " << E_max << std::endl;
    	std::cout << "E_max = " << E_max << std::endl;

    }

    std::cout << "Average time per time step: " << t_total/Nt << " s." << std::endl;
    std::cout << "Total time: " << t_total << " s." << std::endl;


	return 0;
}
