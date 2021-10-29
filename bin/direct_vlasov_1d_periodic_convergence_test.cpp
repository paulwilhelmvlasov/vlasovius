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

#include <omp.h>
#include <cblas.h>
#include <iostream>
#include <armadillo>

#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/kernels/periodised_kernel.h>
#include <vlasovius/interpolators/piecewise_interpolator.h>
#include <vlasovius/interpolators/direct_interpolator.h>
#include <vlasovius/integrators/gauss_konrod.h>
#include <vlasovius/misc/periodic_poisson_1d.h>

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
    return 0.39894228040143267793994
    		* ( (1.0 - nb) * std::exp(-0.5 * v * v)
    				+ 2.0 * nb * std::exp(-2 * (v - vb) * (v - vb)) )
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


struct resolution
{
	size_t Nx;
	size_t Nv;
};


int main()
{
	constexpr size_t order_x = 3;
	constexpr size_t order_v = 3;
	using kernel_t        = vlasovius::xv_kernel<order_x,order_v>;
	using interpolator_t  = vlasovius::interpolators::direct_interpolator<kernel_t>;
	using poisson_t       = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

	using wendland_t = vlasovius::kernels::wendland<1,order_x>;
	wendland_t W;


	constexpr double tikhonov_mu { 1e-10 };
	constexpr size_t min_per_box = 200;

	double L = 4*3.14159265358979323846;

	arma::rowvec sigma { 6.0, 3.0 };
	kernel_t K( sigma[0], sigma[1], L );

	double v_max = 7.5;

	size_t num_threads = omp_get_max_threads();

	resolution test_res {128, 256};

	/*
	std::vector<resolution> res
	{
		{32, 64}, {40, 70}, {50, 80}, {60, 90}, {64, 128}, test_res
	};
	*/

	std::vector<double> h_res
	{
		0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1
	};

	arma::uword N_res_samples = h_res.size() + 1;
	std::vector<resolution> res;

	for(size_t i = 0; i < N_res_samples - 1; ++i)
	{
		res.push_back({size_t(L/h_res[i]), size_t(20/h_res[i])});
	}

	res.push_back(test_res);

	arma::uword test_N = 300;

	std::vector<arma::mat> xv(N_res_samples);
	std::vector<arma::vec> f(N_res_samples);

	for(size_t r = 0; r < N_res_samples; r++)
	{
		arma::uword Nx = res[r].Nx;
		arma::uword Nv = res[r].Nv;
		xv[r].set_size( Nx*Nv,2 );
		f[r].resize( Nx*Nv );
		for ( size_t i = 0; i < Nx; ++i )
			for ( size_t j = 0; j < Nv; ++j )
			{
				double x = i * (L/Nx);
				double v = -v_max + j*(2*v_max/(Nv-1));

				xv[r]( j + Nv*i, 0 ) = x;
				xv[r]( j + Nv*i, 1 ) = v;
				constexpr double alpha = 0.01;
				constexpr double k     = 0.5;
				f[r]( j + Nv*i ) = two_stream_f0(x, v, alpha, k);
			}
	}

	/*
	arma::vec x_test_pts(test_N);
	for(size_t i = 0; i < test_N; i++)
	{
		x_test_pts(i) = i * (L / test_N);
	}
	*/
	size_t Nx_test = 200, Nv_test = 200;
	double hx_test = L / double(Nx_test);
	double hv_test = 20 / double(Nx_test);
	arma::mat xv_test_pts(Nx_test*Nv_test, 2);
	for(size_t i = 0; i < Nx_test; ++i)
		for(size_t j = 0; j < Nv_test; j++)
		{
			double x = i * (L/Nx_test);
			double v = -v_max + j*( 2 * v_max/(Nv_test-1));

			xv_test_pts( j + Nv_test*i, 0 ) = x;
			xv_test_pts( j + Nv_test*i, 1 ) = v;
		}

	poisson_t poisson(0,L,256);
	arma::vec rho_points = poisson.quadrature_nodes();
	arma::vec rho( rho_points.size() );
	vlasovius::geometry::kd_tree rho_tree(rho_points);

	std::vector<interpolator_t> fcts;

	size_t count = 0;
	double t = 0, T = 10, dt = 1./16.;
	//std::ofstream str("E_infty_err.txt");
	std::ofstream str("f_l1_err.txt");
	while ( t < T )
	{
		std::cout << "t = " << t << ". " << std::endl;

		if ( t + dt > T ) dt = T - t;

		arma::mat E_values(test_N, N_res_samples);

		for(size_t r = 0; r < N_res_samples; r++)
		{
			xv[r].col(0) += dt*xv[r].col(1);             // Move particles
			xv[r].col(0) -= L * floor(xv[r].col(0) / L); // Set to periodic positions.
			interpolator_t sfx { K, xv[r], f[r], tikhonov_mu, num_threads };
			rho = arma::vec(rho_points.n_rows,arma::fill::ones)
								        - 2 * W.integral() * sigma[1] * K.eval_x( rho_points, xv[r] ) * sfx.coeffs();
			poisson.update_rho( rho );
			for ( size_t i = 0; i < xv[r].n_rows; ++i )
			{
				double E = poisson.E(xv[r](i,0));
				xv[r](i,1) += -dt*E;
			}

			/*
			for(size_t i = 0; i < test_N; i++)
			{
				E_values(i, r) = poisson.E(x_test_pts(i));
			}
			*/

			fcts.push_back(sfx);
		}

		str << t << " ";
		for(size_t r = 0; r < N_res_samples - 1; r++)
		{
			//double err = abs(E_values.col(r) - E_values.col(N_res_samples - 1)).max();
			// Computing L1-error using mid-point rule:
			double err = hx_test * hv_test * arma::accu(arma::abs(fcts[r](xv_test_pts) - fcts[N_res_samples - 1](xv_test_pts)));
			str << err << " ";
		}
		str << std::endl;

		fcts.clear();

		t += dt;
	}

}
