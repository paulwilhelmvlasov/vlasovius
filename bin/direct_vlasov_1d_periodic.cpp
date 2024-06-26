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
#include <vlasovius/interpolators/direct_interpolator.h>
#include <vlasovius/interpolators/pou_interpolator.h>
#include <vlasovius/misc/periodic_poisson_1d.h>

namespace vlasovius
{

template <size_t k1, size_t k2>
class xv_kernel
{
public:
	xv_kernel() = delete;
	xv_kernel( double sigma_x, double sigma_v, double L );

	arma::mat operator()( const arma::mat &xv1, const arma::mat &xv2 ) const;

	arma::mat eval_x( const arma::mat &xv1, const arma::mat &xv2 ) const;

	void eval( size_t dim, size_t n, size_t m,
			         double *K, size_t ldK,
	           const double *X, size_t ldX,
	           const double *Y, size_t ldY, size_t threads = 1 ) const;

	void mul( size_t dim, size_t n, size_t m, size_t nrhs,
			         double *R, size_t ldK,
	           const double *X, size_t ldX,
	           const double *Y, size_t ldY,
			   const double *C, size_t ldC, size_t threads = 1 ) const;

private:
	kernels::wendland<1,k1> W1;
	kernels::wendland<1,k2> W2;
	double L, inv_sigma_x, inv_sigma_v;
	size_t num_images;
};


template <size_t k1, size_t k2>
xv_kernel<k1,k2>::xv_kernel( double sigma_x, double sigma_v, double L ):
L { L } , inv_sigma_x { 1/sigma_x }, inv_sigma_v { 1/sigma_v },
num_images { static_cast<size_t>(std::ceil(sigma_x/L)) }
{}

template <size_t k1, size_t k2>
arma::mat xv_kernel<k1,k2>::operator()( const arma::mat &xv1, const arma::mat &xv2 ) const
{
	size_t n = xv1.n_rows, m = xv2.n_rows;

	arma::mat result( n,m );
	eval( 2, n, m, result.memptr(), n, xv1.memptr(), n, xv2.memptr(), m );
	return result;
}

template <size_t k1, size_t k2>
arma::mat xv_kernel<k1,k2>::eval_x( const arma::mat &xv1, const arma::mat &xv2 ) const
{
	size_t n = xv1.n_rows, m = xv2.n_rows;
	arma::mat result(n,m);

	using simd_t = ::vlasovius::misc::simd<double>;

	double Linv = 1/L;
	const double *X = xv1.memptr(), *Y = xv2.memptr();
	      double *K = result.memptr();

	#pragma omp parallel for
	for ( size_t j = 0; j < m; ++j )
	{
		size_t i = 0;

		simd_t y; y.fill(Y+j); y = y - L*floor(y*Linv);
		double yy = Y[j]; yy -= L*std::floor(yy*Linv);

		if constexpr ( simd_t::size() > 1 )
		{
			simd_t x, val;

			for ( ; i + simd_t::size() < n; i += simd_t::size()  )
			{
				x.load(X + i);
				x = x - L*floor(x*Linv);

				val = W1( abs(x-y)*inv_sigma_x );
				for ( size_t n = 1; n <= num_images; ++n )
				{
					val = val + W1( abs((x-y) + (n*L))*inv_sigma_x );
					val = val + W1( abs((x-y) - (n*L))*inv_sigma_x );
				}
				val.store( K + i + j*n );
			}
		}

		for ( ; i < n; ++i )
		{
			double xx = X[i]; xx -= L*std::floor(xx*Linv);

			double val = W1( abs(xx-yy)*inv_sigma_x );
			for ( size_t n = 1; n <= num_images; ++n )
			{
				val += W1( abs((xx-yy) + (n*L))*inv_sigma_x );
				val += W1( abs((xx-yy) - (n*L))*inv_sigma_x );
			}
			K[ i + j*n ] = val;
		}
	}

	return result;
}

template <size_t k1, size_t k2>
void xv_kernel<k1,k2>::eval( size_t /* dim */, size_t n, size_t m,
                                   double *K, size_t ldK,
                             const double *X, size_t ldX,
                             const double *Y, size_t ldY, size_t threads ) const
{
	using simd_t = ::vlasovius::misc::simd<double>;

	double Linv = 1/L;

	#pragma omp parallel for if(threads>1),num_threads(threads)
	for ( size_t j = 0; j < m; ++j )
	{
		size_t i = 0;

		simd_t y1; y1.fill(Y+j); y1 = y1 - L*floor(y1*Linv);
		simd_t y2; y2.fill(Y+j+ldY);

		double yy1 = Y[j]; yy1 -= L*std::floor(yy1*Linv);
		double yy2 = Y[j+ldY];

		if constexpr ( simd_t::size() > 1 )
		{
			simd_t x1, x2, val;

			for ( ; i + simd_t::size() < n; i += simd_t::size()  )
			{
				x1.load(X + i);
				x1 = x1 - L*floor(x1*Linv);

				val = W1( abs(x1-y1)*inv_sigma_x );
				for ( size_t n = 1; n <= num_images; ++n )
				{
					val = val + W1( abs((x1-y1) + (n*L))*inv_sigma_x );
					val = val + W1( abs((x1-y1) - (n*L))*inv_sigma_x );
				}

				x2.load(X + i + ldX);
				val = val * W2( abs(x2-y2)*inv_sigma_v );
				val.store( K + i + j*ldK );
			}
		}

		for ( ; i < n; ++i )
		{
			double xx1 = X[i], xx2 = X[i+ldX];

			xx1 -= L*std::floor(xx1*Linv);
			double val = W1( abs(xx1-yy1)*inv_sigma_x );
			for ( size_t n = 1; n <= num_images; ++n )
			{
				val += W1( abs((xx1-yy1) + (n*L))*inv_sigma_x );
				val += W1( abs((xx1-yy1) - (n*L))*inv_sigma_x );
			}

			val *= W2( abs(xx2-yy2)*inv_sigma_v );

			K[ i + j*ldK ] = val;
		}
	}
}

template <size_t k1, size_t k2>
void xv_kernel<k1,k2>::mul( size_t dim, size_t n, size_t m, size_t nrhs,
		                          double *R, size_t ldR,
                            const double *X, size_t ldX,
                            const double *Y, size_t ldY,
	                        const double *C, size_t ldC, size_t threads ) const
{
	arma::mat K( n, m );
	eval( dim, n, m, K.memptr(), n, X, ldX, Y, ldY, threads );
	cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
			     n, nrhs, m, 1.0, K.memptr(), n, C, ldC, 0.0, R, ldR );
}

}

int main()
{
	constexpr size_t order = 2;
	using kernel_t        = vlasovius::xv_kernel<order,order>;
	using interpolator_t  = vlasovius::interpolators::direct_interpolator<kernel_t>;
	using poisson_t       = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

	using wendland_t = vlasovius::kernels::wendland<1,order>;
	wendland_t W;

	double L = 4*3.14159265358979323846, sigma_x  = 4, sigma_v = 2;
	double mu = 1e-12;
	double vmax = 6;
	kernel_t K( sigma_x, sigma_v, L );


	size_t num_threads = 6;//omp_get_max_threads();

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

	size_t res = 256;
	double hx_plot = L / res;
	double hv_plot = 2.0 * vmax / res;
	arma::mat plotX( (res + 1)*(res + 1), 2 );
	arma::vec plotf( (res + 1)*(res + 1));
	for ( size_t i = 0; i <= res; ++i )
		for ( size_t j = 0; j <= res; ++j )
		{
			plotX(j + (res + 1)*i,0) = L * i/double(res);
			plotX(j + (res + 1)*i,1) = 2*vmax * j/double(res) - vmax;
			plotf(j + (res + 1)*i) = 0;
		}

	// Initialise xv.
	size_t Nx = 32, Nv = 128;
	xv.set_size( Nx*Nv,2 );
	f.resize( Nx*Nv );
	for ( size_t i = 0; i < Nx; ++i )
	for ( size_t j = 0; j < Nv; ++j )
	{
		double x = i * (L/Nx);
		double v = -vmax + j*(2*vmax/(Nv-1));

		xv( j + Nv*i, 0 ) = x;
		xv( j + Nv*i, 1 ) = v;
		constexpr double alpha = 0.01;
		constexpr double k     = 0.5;
		// Linear Landau damping:
		f( j + Nv*i ) = 0.39894228040143267793994 * ( 1. + alpha*std::cos(k*x) )
					* std::exp(-0.5 * v * v);


		// Two Stream Instability:
/*
		f( j + Nv*i ) = 0.39894228040143267793994 * ( 1. + alpha*std::cos(k*x) )
						* v * v * std::exp(-0.5 * v * v);
*/

		/*
		// Bump on tail benchmark:
		constexpr double alpha = 0.04;
		constexpr double k     = 0.3;
		constexpr double np = 0.9;
		constexpr double nb = 0.2;
		constexpr double vb = 4.5;
		constexpr double vt = 0.5;
		f( j + Nv*i ) = 0.39894228040143267793994 * ( 1. + alpha*std::cos(k*x) )
				* (np * std::exp( -0.5 * v*v )
				+  nb * std::exp( -0.5 * (v-vb)*(v-vb) / (vt * vt)) );
		*/

	}

	size_t count = 0;
	double totalTime = 0;
	double t = 0, T = 100, dt = 1./16.;
	std::ofstream E_linfty_str("E_linfty.txt");
	std::ofstream E_l2_str("E_l2.txt");
	//std::ofstream str_f_max_err("f_max_error.txt");
	//std::ofstream str_f_l2_err("f_l2_error.txt");
    std::ofstream statistics_file( "statistics.csv" );
    statistics_file << R"("Time"; "L1-Norm"; "L2-Norm"; "Electric Energy"; "Kinetic Energy"; "Total Energy"; "Entropy")";
    statistics_file << std::endl;

    statistics_file << std::scientific;
    std::cout << std::scientific;
    double l1_norm = 0;
    double l2_norm = 0;
    double kinetic_energy = 0;
    double entropy = 0;

	while ( t <= T )
	{
		std::cout << "t = " << t << ". " << std::endl;


		if ( count % (25*16) == 0 )
		{
			interpolator_t sfx { K, xv, f, mu, num_threads };
			plotf = sfx(plotX);
			std::ofstream fstr( "f_" + std::to_string(t) + ".txt" );
			//std::ifstream f_exact_str("../../../PW/res=4096x8192_plotres=400/f_"+ std::to_string(t) + ".txt" );
			//double f_max_error = 0;
			//double f_l2_error = 0;
			for ( size_t i = 0; i <= res; ++i )
			{
				for ( size_t j = 0; j <= res; ++j )
				{
					double x = plotX(j + (res + 1)*i,0);
					double v = plotX(j + (res + 1)*i,1);
					double f = plotf(j+(res+1)*i);
					fstr << x << " " << v << " " << f << std::endl;

					kinetic_energy += v*v*f;
					if(f > 1e-10)
					{
						entropy += f*std::log(f);
					}
					l1_norm += std::abs(f);
					l2_norm += f*f;
				}
				fstr << "\n";
			}

			kinetic_energy *= 0.5*hx_plot*hv_plot;
			entropy *= hx_plot*hv_plot;
			l1_norm *= hx_plot*hv_plot;
			l2_norm = hx_plot*hv_plot*std::sqrt(l2_norm);
		}



		vlasovius::misc::stopwatch clock;
		for ( size_t stage = 0; stage < 4; ++stage )
		{
			arma::mat xv_stage = xv;
			for ( size_t s = 0;  s+1 <= stage; ++s )
				xv_stage += dt*c_rk4[stage][s]*k_xv[s];

			k_xv[ stage ].resize( xv.n_rows, xv.n_cols );
			k_xv[ stage ].col(0) = xv_stage.col(1);

			interpolator_t sfx( K, xv_stage, f, mu, num_threads );
			arma::vec rho = arma::vec(rho_points.n_rows,arma::fill::ones) -
					        2 * W.integral() * sigma_v * K.eval_x( rho_points, xv_stage ) * sfx.coeffs();
			poisson.update_rho( rho );

			for ( size_t i = 0; i < xv_stage.n_rows; ++i )
				k_xv[stage](i,1) = -poisson.E( xv_stage(i,0) );

			if ( stage == 0 )
			{
				E_linfty_str << t << " " << norm(k_xv[stage].col(1),"inf")  << std::endl;
				std::cout << "Max-norm of E: " << norm(k_xv[stage].col(1),"inf") << "." << std::endl;

				size_t plot_x = 256;
				double dx_plot = L/plot_x;
				double E_l2_norm = 0;
				for(size_t i = 0; i < plot_x; i++)
				{
					double x = (i+0.5)*dx_plot;
					double E = poisson.E(x);
					E_l2_norm += E*E;
				}
				E_l2_norm *= 0.5*dx_plot;
				E_l2_str << t << " " << E_l2_norm << std::endl;

				/*
				if ( count % (10*16) == 0 )
				{
		            statistics_file << t       << "; "
		                            << l1_norm      << "; "
		                            << l2_norm      << "; "
		                            << E_l2_norm << "; "
		                            << kinetic_energy << "; "
		                            << E_l2_norm + kinetic_energy << "; "
		                            << entropy << std::endl;

		            l1_norm = 0;
		            l2_norm = 0;
		            kinetic_energy = 0;
		            entropy = 0;
				}
				*/
			}
 		}



		for ( size_t s = 0; s < 4; ++s )
			xv += dt*d_rk4[s]*k_xv[s];
		t += dt;

		double elapsed = clock.elapsed();
		count++;
		totalTime += elapsed;

		std::cout << "Time for needed for time-step: " << elapsed << ".\n";
		std::cout << "---------------------------------------" << elapsed << ".\n";

		//if ( t + dt > T ) dt = T - t;
	}

	std::cout << "Total simulation time: " << totalTime << std::endl;
	std::cout << "Average time for a time-step: " << totalTime / count << std::endl;


}
