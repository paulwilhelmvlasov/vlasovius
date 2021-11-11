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

double bump_on_tail_f0(double x, double v, double alpha = 0.01, double k = 0.5, double np = 0.9,
		double nb = 0.2, double vb = 4.5, double vt = 0.5)
{
    return 0.39894228040143267793994 *
    	(np * std::exp(-0.5 * v * v) + nb * std::exp(-0.5 * (v - vb) * (v - vb) / (vt * vt)))
        * (1.0 + alpha * std::cos(k * x));

}

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
	constexpr size_t order_x = 4;
	constexpr size_t order_v = 4;
	using kernel_t        = vlasovius::xv_kernel<order_x,order_v>;
	using interpolator_t  = vlasovius::interpolators::direct_interpolator<kernel_t>;
	using poisson_t       = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

	using wendland_t = vlasovius::kernels::wendland<1,order_x>;
	wendland_t W;

	size_t res_n = 100;

	double L = 2*3.14159265358979323846 / 0.3, sigma_x  = 2, sigma_v = 2;
	double mu = 1e-10;
	double v_max = 8;
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
	size_t Nx = 32, Nv = 64;
	xv.set_size( Nx*Nv,2 );
	f.resize( Nx*Nv );
	for ( size_t i = 0; i < Nx; ++i )
	for ( size_t j = 0; j < Nv; ++j )
	{
		double x = i * (L/Nx);
		double v = -v_max + j*(2.0 * v_max/(Nv-1));

		xv( j + Nv*i, 0 ) = x;
		xv( j + Nv*i, 1 ) = v;
		constexpr double alpha = 0.04;
		constexpr double K     = 0.3;
		//f( j + Nv*i ) = lin_landau_f0(x, v, alpha, K);
		//f( j + Nv*i ) = two_stream_f0(x, v, alpha, K);
		f( j + Nv*i ) = bump_on_tail_f0(x, v, alpha, K);
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
	double t = 0, T = 500.25, dt = 1./4.;
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
							//<< f - maxwellian(v) << std::endl;
							<< f << std::endl;
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
