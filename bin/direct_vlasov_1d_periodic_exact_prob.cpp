/*
 * Copyright (C) 2021 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of vlasovius.
 *
 * vlasovius is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * vlasovius is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * vlasovius; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */

#include <cblas.h>
#include <iostream>
#include <armadillo>

#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/kernels/periodised_kernel.h>
#include <vlasovius/interpolators/direct_interpolator.h>
#include <vlasovius/interpolators/pou_interpolator.h>
#include <vlasovius/misc/periodic_poisson_1d.h>
#include <vlasovius/misc/xv_kernel.h>

double maxwellian(double v)
{
	return 0.39894228040143267793994 * std::exp(-0.5 * v * v);
}

double lin_landau_f0(double x, double v, double alpha = 0.01, double k = 0.5)
{
	return 0.39894228040143267793994 * ( 1 + alpha * std::cos(k * x) )
			* std::exp( -0.5 * v*v );
}

double two_stream_f0(double x, double v, double alpha = 0.01, double k = 0.5)
{
    return 0.39894228040143267793994 * v * v * std::exp(-0.5 * v * v )
	        * (1.0 + alpha * std::cos(k * x));
}

double bump_on_tail_f0(double x, double v, double alpha = 0.01, double k = 0.5, double nb = 0.1, double vb = 4.5)
{
    return 0.39894228040143267793994 * ((1.0 - nb) * std::exp(-0.5 * v * v)
        + nb * std::exp(-0.5 * (v - vb) * (v - vb)))
        * (1.0 + alpha * std::cos(k * x));
}

namespace vlasovius
{
namespace interpolators
{

vlasovius::kernels::wendland<1,2> someKernel;
double C = someKernel.integral();

template <typename kernel>
class dep_interpolator
{
public:
	dep_interpolator() = default;
	dep_interpolator( const dep_interpolator&  )  = default;
	dep_interpolator(       dep_interpolator&&  ) = default;

	dep_interpolator& operator=( const dep_interpolator&  ) = default;
	dep_interpolator& operator=(       dep_interpolator&& ) = default;

	dep_interpolator( kernel K, arma::mat X, arma::mat b,
			             double tikhonov_mu = 0, size_t threads = 1 );

	arma::mat operator()( const arma::mat &Y, size_t threads = 1 ) const;
	const arma::mat& coeffs() const noexcept { return coeff; }
	const arma::mat& points() const noexcept { return X; }

private:
	kernel    K;
	arma::mat X;
	arma::mat coeff;
};

template <typename kernel>
dep_interpolator<kernel>::dep_interpolator(kernel K, arma::mat X,
		arma::mat b, double tikhonov_mu, size_t threads)
{
	size_t dim = X.n_cols; size_t N = X.n_cols;
	arma::mat kermat(N, N);
	K.eval(dim, N, N, &kermat(0,0), N, &X(0,0), N, &X(0,0), N, threads);

	kermat.row(N - 1) = arma::rowvec(N, arma::fill::ones);

	arma::colvec rhs = b;
	rhs(N-1) = 4*3.14159265358979323846 / (C * C);

	coeff = solve(kermat, rhs);
}

template <typename kernel>
arma::mat dep_interpolator<kernel>::operator()( const arma::mat &Y, size_t threads ) const
{
	if ( Y.empty() )
	{
		throw std::logic_error { "vlasovius::dep_interpolator::operator(): "
				                 "No evaluation points passed." };
	}

	if ( X.n_cols != Y.n_cols )
	{
		throw std::logic_error { "vlasovius::dep_interpolator::operator(): "
		                         "Evaluation and Interpolation points have differing dimensions." };
	}

	size_t dim = X.n_cols, n = Y.n_rows, m = X.n_rows;
	arma::mat result( Y.n_rows, coeff.n_cols, arma::fill::zeros );

	K.mul( dim, n, m, coeff.n_cols, result.memptr(), n,
                                         Y.memptr(), n,
									     X.memptr(), m,
	                                 coeff.memptr(), m, threads );
	return result;
}

}
}

int main()
{
	constexpr size_t order_x = 2;
	constexpr size_t order_v = 2;
	using kernel_t        = vlasovius::xv_kernel<order_x,order_v>;
	//using interpolator_t  = vlasovius::interpolators::direct_interpolator<kernel_t>;
	using interpolator_t  = vlasovius::interpolators::dep_interpolator<kernel_t>;
	using poisson_t       = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

	using wendland_t = vlasovius::kernels::wendland<1,order_x>;
	wendland_t W;

	size_t res_n = 400;

	double L = 4*3.14159265358979323846, sigma_x  = 2, sigma_v = 2;
	double mu = 1e-10;
	double v_max = 6;
	kernel_t K( sigma_x, sigma_v, L );


	size_t num_threads = omp_get_max_threads();

	// Runge--Kutta Butcher tableau.
	constexpr double c_rk4[4][4] = { {   0,   0,  0, 0 },
	                                 { 0.5,   0,  0, 0 },
	                                 {   0, 0.5,  0, 0 },
	                                 {   0,   0,  1, 0 } };
	constexpr double d_rk4[4] = { 1./6., 1./3., 1./3., 1./6. };


	arma::mat xv, k_xv[ 4 ];
	arma::vec f;

	poisson_t poisson(0,L,256);
	arma::mat rho_points;
	{
		arma::vec rho_tmp = poisson.quadrature_nodes();
		rho_points.set_size( rho_tmp.n_rows, 2 );
		for ( size_t i = 0; i < rho_tmp.size(); ++i )
		{
			rho_points(i,0) = rho_tmp(i);
			rho_points(i,1) = 0;
		}
	}

	// Initialise xv.
	size_t Nx = 32, Nv = 128;
	xv.set_size( Nx*Nv,2 );
	f.resize( Nx*Nv );
	for ( size_t i = 0; i < Nx; ++i )
	for ( size_t j = 0; j < Nv; ++j )
	{
		double x = i * (L/Nx);
		double v = -v_max + j*(2.0 * v_max/(Nv-1));

		xv( j + Nv*i, 0 ) = x;
		xv( j + Nv*i, 1 ) = v;
		constexpr double alpha = 0.01;
		constexpr double K     = 0.5;
		f( j + Nv*i ) = lin_landau_f0(x, v, alpha, K);
	}

	arma::mat plotX( (res_n + 1)*(res_n + 1), 2 );
	arma::vec plotf( (res_n + 1)*(res_n + 1) );
	for ( size_t i = 0; i <= res_n; ++i )
		for ( size_t j = 0; j <= res_n; ++j )
		{
			plotX(j + (res_n + 1)*i,0) = L * i/double(res_n);
			plotX(j + (res_n + 1)*i,1) = 2*v_max * j/double(res_n) - v_max;
			plotf(j + (res_n + 1)*i) = 0;
		}

	// t == 0 plot:
	interpolator_t sfx_plot( K, xv, f, mu, num_threads );
	plotf = sfx_plot(plotX);
		std::ofstream fstr( "f_" + std::to_string(0.0) + "s.txt" );
		for ( size_t i = 0; i <= res_n; ++i )
		{
			for ( size_t j = 0; j <= res_n; ++j )
			{
				arma::uword index = j + (res_n + 1)*i;
				fstr << plotX(index,0) << " " << plotX(index,1)
					 << " " << plotf(index) - maxwellian(plotX(index,1))
					 << std::endl;
			}
			fstr << "\n";
		}


	size_t count = 0;
	vlasovius::misc::stopwatch main_clock;
	double t = 0, T = 50.25, dt = 1./4.;
	std::ofstream str("E.txt");
	while ( t < T )
	{
		std::cout << "t = " << t << ". " << std::endl;
		vlasovius::misc::stopwatch clock;
		for ( size_t stage = 0; stage < 4; ++stage )
		{
			arma::mat xv_stage = xv;
			for ( size_t s = 0;  s+1 <= stage; ++s )
				xv_stage += dt*c_rk4[stage][s]*k_xv[s];

			k_xv[ stage ].resize( xv.n_rows, xv.n_cols );
			k_xv[ stage ].col(0) = xv_stage.col(1);

			interpolator_t sfx( K, xv_stage, f, mu, num_threads );
			arma::vec rho = arma::vec(rho_points.n_rows,arma::fill::ones)
					        - 2 * W.integral() * sigma_v * K.eval_x( rho_points, xv_stage ) * sfx.coeffs();
			poisson.update_rho( rho );

			for ( size_t i = 0; i < xv_stage.n_rows; ++i )
				k_xv[stage](i,1) = -poisson.E( xv_stage(i,0) );

			if ( stage == 0 )
			{
				str << t << " " << norm(k_xv[stage].col(1),"inf")  << std::endl;
				std::cout << "Max-norm of E: " << norm(k_xv[stage].col(1),"inf") << "." << std::endl;
			}
 		}

		for ( size_t s = 0; s < 4; ++s )
			xv += dt*d_rk4[s]*k_xv[s];
		t += dt;

		double elapsed = clock.elapsed();
		std::cout << "Time for needed for time-step: " << elapsed << ".\n";
		std::cout << "---------------------------------------" << elapsed << ".\n";

		interpolator_t sfx_plot( K, xv, f, mu, num_threads );
		plotf = sfx_plot(plotX);
		if ( ++count % 4 == 0)
		{
			std::ofstream fstr( "f_" + std::to_string(t) + "s.txt" );
			for ( size_t i = 0; i <= res_n; ++i )
			{
				for ( size_t j = 0; j <= res_n; ++j )
				{
					arma::uword index = j + (res_n + 1)*i;
					double x = plotX(index,0);
					double v = plotX(index,1);
					double f = plotf(index);
					fstr << x << " " << v << " "
							<< f - maxwellian(v) << std::endl;
					/*
					if(f < 0)
						fstr << 0 << std::endl;
					else
						fstr << f << std::endl;
					*/
				}
				fstr << "\n";
			}
		}

		if ( t + dt > T ) dt = T - t;
	}

	double main_elapsed = main_clock.elapsed();
	std::cout << "Time for needed for simulation: " << main_elapsed << ".\n";

}

