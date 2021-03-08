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

#include <iostream>
#include <armadillo>

#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/kernels/periodised_kernel.h>
#include <vlasovius/integrators/gauss_konrod.h>
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
			   double *__restrict__ K, size_t ldK,
	           const double        *X, size_t ldX,
	           const double        *Y, size_t ldY ) const;

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
                             double *__restrict__ K, size_t ldK,
                             const double        *X, size_t ldX,
                             const double        *Y, size_t ldY ) const
{
	using simd_t = ::vlasovius::misc::simd<double>;

	double Linv = 1/L;

	#pragma omp parallel for
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

}


int main()
{
	constexpr size_t dim { 2 }, k { 4 };
	constexpr size_t N { 1'000'000 };
	constexpr double tikhonov_mu { 1e-12 };
	constexpr size_t min_per_box = 100;
	constexpr double enlarge = 1.5;
	constexpr double twopi { 2*3.1415926535 };

	constexpr double L { 4.0 * 3.1415926535};

	std::cout << "N = " << N << std::endl;
	std::cout << "min per Box = " << min_per_box << std::endl;

	constexpr size_t order = 4;

	using wendland_t     = vlasovius::kernels::wendland<dim,k>;
	//using kernel_t       = vlasovius::xv_kernel<order,4>;
	using kernel_t		 = vlasovius::kernels::rbf_kernel<wendland_t>;
	using interpolator_t = vlasovius::interpolators::pou_interpolator<kernel_t>;

	arma::mat X( N, 2, arma::fill::randu );
	X.col(0) = L * X.col(0);
	X.col(1) = 20.0 * X.col(1) - 10.0;
	arma::vec f( N );
	for ( size_t i = 0; i < N; ++i )
	{
		double x = X(i,0);
		double y = X(i,1);
		f(i) = 0.39894228040143267793994 * ( 1 + 0.01*std::cos(0.5*x) )
				* std::exp( - y * y /2 );
	}

	arma::rowvec bounding_box { 0, 0, 1, 1 };
	vlasovius::misc::stopwatch clock;
	kernel_t K { wendland_t {}, 0.5 };
	interpolator_t sfx { K, X, f, bounding_box, enlarge, min_per_box, tikhonov_mu };
	double elapsed { clock.elapsed() };
	std::cout << "Time for computing RBF-Approximation: " << elapsed << ".\n";
	clock.reset();
	double error = norm(f-sfx(X),"inf");
	elapsed = clock.elapsed();
	std::cout << "Maximal interpolation error: " << error << ".\n";
	std::cout << "Time for evaluating interpolation error: " << elapsed << ".\n";


	arma::mat plotX( 1001*1001, 2 );
	arma::vec plotf_true( 1001*1001 );
	for ( size_t i = 0; i <= 1000; ++i )
		for ( size_t j = 0; j <= 1000; ++j )
		{
			double x = plotX(j + 1001*i,0) = i/1000.;
			double y = plotX(j + 1001*i,1) = j/1000.;
			plotf_true(j + 1001*i) = std::sin(twopi*x)*std::sin(twopi*y);
		}

	clock.reset();
	arma::vec plotf = sfx(plotX);
	elapsed = clock.elapsed();
	std::cout << "Time for evaluating RBF-approximation at plotting points: " << elapsed << ".\n";
	std::cout << "Maximum encountered error at plotting points: " << norm(plotf-plotf_true,"inf") << ".\n";

	std::ofstream str( "test_pou_interpolator.txt" );
	for ( size_t i = 0; i <= 1000; ++i )
	{
		for ( size_t j = 0; j <= 1000; ++j )
		{
			double x = plotX(j + 1001*i,0);
			double y = plotX(j + 1001*i,1);
			double err  = plotf(j+1001*i)-plotf_true(j+1001*i);
			str << x << " " << y << " " << err << std::endl;
		}
		str << "\n";
	}


	return 0;
}
