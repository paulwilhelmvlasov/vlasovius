//============================================================================
// Name        : Particle-In-Cell.cpp
// Author      : Rostislav-Paul Wilhelm
// Version     :
// Copyright   : This is a custom PIC code to solve Vlasov-Poisson. It goes under a GNU-GPL.
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

    //return fac*(1+alpha*cos(k*x))*exp(-v*v/2) * v*v;
    return fac*(1+alpha*cos(k*x))*exp(-v*v/2);
}

inline double f0_1d_electron(double x, double v)
{
    using std::cos;
    using std::exp;
    constexpr double alpha = 0.01;
    constexpr double k     = 0.5;
    constexpr double Ue = -2;
    constexpr double fac   = 0.39894228040143267793994;

    return fac * ( 1. + alpha*std::cos(k*x) ) * std::exp( -(v-Ue)*(v-Ue)/2. );
}


inline double f0_1d_ion(double x, double v)
{
    using std::cos;
    using std::exp;
    constexpr double alpha = 0.01;
    constexpr double k     = 0.5;
    constexpr double Mr = 1000;
    constexpr double Ue = -2;
    constexpr double fac   = 12.61566261010080024123574761182842;

    return fac * std::exp( -Mr*v*v/2. );
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

inline double w_3(double x, double eps = 1)
{
	x = x/eps;
	// 2nd order standard univariate B-Spline:
	if(x < 0)
	{
		x = -x;
	}

	if(x > 2)
	{
		return 0;
	}
	if(x < 1)
	{
		return (1 - 0.5*x - x*x + 0.5*x*x*x) / eps;
	}

	return (1 - 11.0/6.0*x + x*x - x*x*x/6.0) / eps;
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

double eval_f(double x, double v, const arma::mat& xv, const arma::vec& Q, double delta_x, double delta_v)
{
	double f = 0;
	for(size_t i = 0; i < xv.n_rows; i++)
	{
		f += Q(i)*shape_function_1d( x - xv(i,0), delta_x)*shape_function_1d( v - xv(i,1), delta_v);
	}

	return f;
}

void ion_acoustic()
{
	// This is an implementation of Wang et.al. 2nd-order PIC (see section 3.2).
	// Set parameters.
    const double L  = 4*3.14159265358979323846;
    const size_t Nx_f = 128;
    const size_t Nx_poisson = Nx_f/2;
    const size_t Nv_f_electron = 512;
    const size_t Nv_f_ion = Nv_f_electron;
    const size_t N_f_electron = Nx_f*Nv_f_electron;
    const size_t N_f_ion = Nx_f*Nv_f_ion;
    const double v_min_electron = -8;
    const double v_max_electron =  4;
    const double v_min_ion = -0.2;
    const double v_max_ion =  0.2;

    constexpr double alpha = 0.01;
    constexpr double k     = 0.5;
    constexpr double Mr	 = 1000; // (approximate) mass ratio between electron and ions.
    constexpr double Ue 	 = -2;

    // Compute derived quantities.
    const double eps_x = L/Nx_f;
    const double eps_v_electron = (v_max_electron-v_min_electron)/Nv_f_electron;
    const double eps_v_ion = (v_max_ion-v_min_ion)/Nv_f_ion;
    const double delta_x = L/Nx_poisson;
    const double delta_x_inv = 1/delta_x;
    const double L_inv  = 1/L;

    const size_t Nt = 100 * 16;
    const double dt = 1.0 / 16.0;

    // Initiate particles.
    // Electrons:
    arma::mat xv_electron;
    arma::vec Q_electron(N_f_electron);
    xv_electron.set_size(N_f_electron,2);

    for ( size_t i = 0; i < Nx_f; ++i )
    for ( size_t j = 0; j < Nv_f_electron; ++j )
    {
        double x = (i+0.5)*eps_x;
        double v = v_min_electron + (j+0.5)*eps_v_electron;
        size_t k = j+Nv_f_electron*i;
        xv_electron(k, 0 ) = x;
        xv_electron(k, 1 ) = v;

        Q_electron(k) = eps_x*eps_v_electron*f0_1d_electron(x,v);
    }

    // Ions:
    arma::mat xv_ion;
    arma::vec Q_ion(N_f_ion);
    xv_ion.set_size(N_f_ion,2);

    for ( size_t i = 0; i < Nx_f; ++i )
    for ( size_t j = 0; j < Nv_f_ion; ++j )
    {
        double x = (i+0.5)*eps_x;
        double v = v_min_ion + (j+0.5)*eps_v_ion;
        size_t k = j+Nv_f_ion*i;
        xv_ion(k, 0 ) = x;
        xv_ion(k, 1 ) = v;

        Q_ion(k) = eps_x*eps_v_ion * f0_1d_ion(x,v);
    }

    arma::vec rho(Nx_poisson + 1);
    arma::vec rho_ion(Nx_poisson + 1);
    arma::vec rho_electron(Nx_poisson + 1);
    arma::vec varphi(Nx_poisson + 1);
    arma::vec E(Nx_poisson + 1);

    arma::mat poiss_solve_matrix(Nx_poisson + 1, Nx_poisson + 1, arma::fill::zeros);
    init_poiss_matrix(poiss_solve_matrix, delta_x);

    // Time-loop using symplectic Euler.
    std::ofstream E_max_str("E_max.txt");
    std::ofstream E_l2_str("E_l2.txt");
    double t_total = 0;
    for(size_t nt = 0; nt <= Nt; nt++)
    {
        vlasovius::misc::stopwatch clock;

    	// Compute electron density.
		#pragma omp parallel for
        for(size_t i = 1; i < Nx_poisson; i++)
    	{
    		double x = i*delta_x;
    		rho_ion(i) = 0;

    		for(size_t k = 0; k < N_f_ion; k++)
    		{
    			rho_ion(i) += Q_ion(k) * shape_function_1d( x - xv_ion(k,0), delta_x);
    		}
    	}
		#pragma omp parallel for
		for(size_t i = 1; i < Nx_poisson; i++)
		{
			double x = i*delta_x;
			rho_electron(i) = 0;

    		for(size_t k = 0; k < N_f_electron; k++)
    		{
    			rho_electron(i) += Q_electron(k) * shape_function_1d( x - xv_electron(k,0), delta_x);
    		}
		}

		rho = rho_ion - rho_electron;
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

    	// Move electron particles.
		#pragma omp parallel for
    	for(size_t k = 0; k < N_f_electron; k++ )
    	{
    		xv_electron(k, 1) -= dt * eval_E(xv_electron(k,0), E, delta_x);
    		xv_electron(k, 0) += dt * xv_electron(k,1);
    		xv_electron(k, 0) -= L*std::floor(xv_electron(k, 0)*L_inv);
    	}

    	// Move ion particles.
		#pragma omp parallel for
    	for(size_t k = 0; k < N_f_ion; k++ )
    	{
    		xv_ion(k, 1) += dt * eval_E(xv_ion(k,0), E, delta_x);
    		xv_ion(k, 0) += dt * xv_ion(k,1);
    		xv_ion(k, 0) -= L*std::floor(xv_ion(k, 0)*L_inv);
    	}

    	double t = nt * dt;
    	double elapsed = clock.elapsed();
    	t_total += elapsed;
    	std::cout << t << "  " << elapsed << " " << t_total << " ";

    	// Do analytics.
    	double E_max = E.max();
    	double E_l2 = 0;
    	for(size_t i = 0; i < Nx_poisson; i++)
    	{
    		E_l2 += E(i)*E(i);
    	}
    	E_l2 *= 0.5*L/Nx_poisson;
    	E_max_str << t << " " << E_max << std::endl;
    	E_l2_str << t << " " << E_l2 << std::endl;
    	std::cout << "E_max = " << E_max << std::endl;

    	if(nt % (16) == 0 && true)
    	{
    		size_t plot_x = 256;
    		size_t plot_v = plot_x;
    		double v_min_plot_electron = -10;
    		double v_max_plot_electron = 10;
    		double v_min_plot_ion = -2;
    		double v_max_plot_ion = 2;
    		double dx_plot = L/plot_x;
    		double dv_plot_electron = (v_max_plot_electron-v_min_plot_electron)/plot_v;
    		double dv_plot_ion = (v_max_plot_ion-v_min_plot_ion)/plot_v;

    		std::ofstream E_str("E_" + std::to_string(t) + ".txt");
    		for(size_t i = 0; i <= plot_x; i++)
    		{
    			double x = i*dx_plot;
    			double value = eval_E(x, E, delta_x);

    			E_str << x << " " << value << std::endl;
    		}

    		/*
    		std::ofstream f_electron_str("f_electon_" + std::to_string(t) + ".txt");
    		for(size_t i = 0; i <= plot_x; i++)
    		{
    			for(size_t j = 0; j <= plot_v; j++)
    			{
    				double x = i*dx_plot;
    				double v = v_min_plot_electron +  j*dv_plot_electron;

    				double f = eval_f(x, v, xv_electron, Q_electron, eps_x, eps_v_electron);

    				f_electron_str << x << " " << v << " " << f << std::endl;
    			}
    			f_electron_str << std::endl;
    		}

    		std::ofstream f_ion_str("f_ion_" + std::to_string(t) + ".txt");
    		for(size_t i = 0; i <= plot_x; i++)
    		{
    			for(size_t j = 0; j <= plot_v; j++)
    			{
    				double x = i*dx_plot;
    				double v = v_min_plot_ion +  j*dv_plot_ion;

    				double f = eval_f(x, v, xv_ion, Q_ion, eps_x, eps_v_ion);

    				f_ion_str << x << " " << v << " " << f << std::endl;
    			}
    			f_ion_str << std::endl;
    		}
    		*/
    	}
    }

    std::cout << "Average time per time step: " << t_total/Nt << " s." << std::endl;
    std::cout << "Total time: " << t_total << " s." << std::endl;
}

void single_species()
{
	// This is an implementation of Wang et.al. 2nd-order PIC (see section 3.2).
	// Set parameters.
    const double L  = 4*3.14159265358979323846;
    const size_t Nx_f = 128;
    const size_t Nx_poisson = Nx_f/2;
    const size_t Nv_f = 256;
    const size_t N_f = Nx_f*Nv_f;
    const double v_min = -6;
    const double v_max =  6;

    // Compute derived quantities.
    const double eps_x = L/Nx_f;
    const double eps_v = (v_max-v_min)/Nv_f;
    const double delta_x = L/Nx_poisson;
    const double delta_x_inv = 1/delta_x;
    const double L_inv  = 1/L;

    const size_t Nt = 0;//30 * 16;
    const double T = 30;
    const double dt = 1.0 / 16.0;

    const size_t n_remap = 5;
    const bool remap = true;

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
        //Q(k)     = eps_x*eps_v*f0_1d(x,v);
        Q(k)     = eps_x*eps_v*f0_1d_electron(x,v);
    }

    arma::vec rho(Nx_poisson + 1);
    arma::vec varphi(Nx_poisson + 1);
    arma::vec E(Nx_poisson + 1);

    arma::mat poiss_solve_matrix(Nx_poisson + 1, Nx_poisson + 1, arma::fill::zeros);
    init_poiss_matrix(poiss_solve_matrix, delta_x);

    //std::cout << poiss_solve_matrix << std::endl;

    // Time-loop using symplectic Euler.
    std::ofstream E_max_str("E_max.txt");
    std::ofstream E_l2_str("E_l2.txt");
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

        std::ofstream rho_electron_str("rho_electron.txt");
        for(size_t i = 1; i < Nx_poisson; i++)
    	{
    		double x = i*delta_x;
    		rho_electron_str << x << " " << rho(i) << std::endl;
    	}

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

    	// Remapping.
    	/*
    	if( remap && (nt % n_remap == 0) && (nt != 0) )
    	{
			arma::vec Q_remap = Q;
			arma::mat xv_remap = xv;
			#pragma omp parallel for
			for ( size_t i = 0; i < Nx_f; ++i )
			for ( size_t j = 0; j < Nv_f; ++j )
			{
				double x =         (i+0.5)*eps_x;
				double v = v_min + (j+0.5)*eps_v;
				size_t p = j+Nv_f*i;
				Q_remap(p) = 0;
				for(size_t k = 0; k < N_f; k++)
				{
					//Q_remap(p) += Q(k) * shape_function_1d(x - xv(k,0), eps_x) * shape_function_1d(v - xv(k,1), eps_v);
					Q_remap(p) += Q(k) * w_3(x - xv(k,0), eps_x) * w_3(v - xv(k,1), eps_v);
				}
				xv_remap(p, 0) = x;
				xv_remap(p, 1) = v;
			}
			Q = eps_x * eps_v * Q_remap;
			xv = xv_remap;
    	}
    	*/

    	double t = nt * dt;
    	double elapsed = clock.elapsed();
    	t_total += elapsed;
    	std::cout << t << "  " << elapsed << " ";

    	// Do analytics.
    	double E_max = E.max();
    	double E_l2 = 0;
    	for(size_t i = 0; i < Nx_poisson; i++)
    	{
    		E_l2 += E(i)*E(i);
    	}
    	E_l2 *= 0.5*L/Nx_poisson;
    	E_max_str << t << " " << E_max << std::endl;
    	E_l2_str << t << " " << E_l2 << std::endl;
    	std::cout << "E_max = " << E_max << std::endl;

    	if(nt % (10*16) == 0)
    	{
    		size_t plot_x = 256;
    		size_t plot_v = plot_x;
    		double dx_plot = L/plot_x;
    		double dv_plot = (v_max-v_min)/plot_v;

    		std::ofstream E_str("E_" + std::to_string(t) + ".txt");
    		for(size_t i = 0; i <= plot_x; i++)
    		{
    			double x = i*dx_plot;
    			double value = eval_E(x, E, delta_x);

    			E_str << x << " " << value << std::endl;
    		}


    		std::ofstream f_str("f_" + std::to_string(t) + ".txt");
    		for(size_t i = 0; i <= plot_x; i++)
    		{
    			for(size_t j = 0; j <= plot_v; j++)
    			{
    				double x = i*dx_plot;
    				double v = v_min +  j*dv_plot;

    				double f = eval_f(x, v, xv, Q, eps_x, eps_v);

    				f_str << x << " " << v << " " << f << std::endl;
    			}
    			f_str << std::endl;
    		}
    	}
    }

    //std::cout << "Average time per time step: " << t_total/Nt << " s." << std::endl;
    std::cout << "Total time: " << t_total << " s." << std::endl;
}

int main() {
	//single_species();
	ion_acoustic();

	return 0;
}
