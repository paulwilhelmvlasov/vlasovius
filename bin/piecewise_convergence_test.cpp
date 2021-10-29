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
#include <iostream>
#include <armadillo>

#include <vlasovius/misc/stopwatch.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/tensorised_kernel.h>
#include <vlasovius/interpolators/piecewise_interpolator.h>
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

struct resolution
{
	size_t Nx;
	size_t Nv;
};

constexpr size_t order = 1;
using wendland_t       = vlasovius::kernels::wendland<1,order>;
using kernel_t         = vlasovius::kernels::tensorised_kernel<wendland_t>;
using interpolator_t   = vlasovius::interpolators::piecewise_interpolator<kernel_t>;
using poisson_t        = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

int main()
{
	constexpr double tikhonov_mu { 1e-10 };
	constexpr size_t min_per_box = 200;

	double L = 4*3.14159265358979323846;

	wendland_t W;
	arma::rowvec sigma { 6.0, 3.0 };
	kernel_t   K ( W, sigma );
	kernel_t   Kx( W, arma::rowvec { sigma(0) } );

	double v_max = 10;

	size_t num_threads = omp_get_max_threads();

	resolution test_res {1024, 4096};

	std::vector<resolution> res
	{
		{32, 64}, {40, 70}, {50, 80}, {60, 90}, {64, 128}, {96, 180}, {128, 256}, {256, 512}, {512, 1024}, test_res
	};

	arma::uword N_res_samples = res.size();
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

	arma::vec x_test_pts(test_N);
	for(size_t i = 0; i < test_N; i++)
	{
		x_test_pts(i) = i * (L / test_N);
	}

	poisson_t poisson(0,L,256);
	arma::vec rho_points = poisson.quadrature_nodes();
	arma::vec rho( rho_points.size() );
	vlasovius::geometry::kd_tree rho_tree(rho_points);

	size_t count = 0;
	double t = 0, T = 10, dt = 1./16.;
	std::ofstream str("E_infty_err.txt");
	while ( t < T )
	{
		std::cout << "t = " << t << ". " << std::endl;

		if ( t + dt > T ) dt = T - t;

		arma::mat E_values(test_N, N_res_samples);

		for(size_t r = 0; r < N_res_samples; r++)
		{
			xv[r].col(0) += dt*xv[r].col(1);             // Move particles
			xv[r].col(0) -= L * floor(xv[r].col(0) / L); // Set to periodic positions.
			interpolator_t sfx { K, xv[r], f[r], min_per_box, tikhonov_mu, num_threads };
			rho.fill(1);
			#pragma omp parallel
			{
				arma::vec my_rho(rho.size(),arma::fill::zeros);
				#pragma omp for schedule(dynamic)
				for ( size_t i = 0; i < sfx.cover.n_rows; ++i )
				{
					const auto &local = sfx.local_interpolants[i];
					double x_min = sfx.cover(i,0), v_min = sfx.cover(i,1),
						   x_max = sfx.cover(i,2), v_max = sfx.cover(i,3);

					const arma::mat &X = local.points();
					arma::vec coeff = local.coeffs();
					for ( size_t j = 0; j < coeff.size(); ++j )
					{
						double v = X(j,1);
						coeff(j) *= ( W.integral((v-v_min)/sigma(1)) +
						              W.integral((v_max-v)/sigma(1)) )*sigma(1);
					}

					arma::rowvec box { x_min, x_max };
					arma::uvec idx = rho_tree.index_query(box);
					my_rho(idx) += Kx(rho_points(idx),X)*coeff;
				}

				#pragma omp critical
				rho -= my_rho;
			}

			poisson.update_rho( rho );
			for ( size_t i = 0; i < xv[r].n_rows; ++i )
			{
				double E = poisson.E(xv[r](i,0));
				xv[r](i,1) += -dt*E;
			}

			for(size_t i = 0; i < test_N; i++)
			{
				E_values(i, r) = poisson.E(x_test_pts(i));
			}
		}

		str << t << " ";
		for(size_t r = 0; r < N_res_samples - 1; r++)
		{
			double err = abs(E_values.col(r) - E_values.col(N_res_samples - 1)).max();
			str << err << " ";
		}
		str << std::endl;

		t += dt;
	}

}
