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
#include <vlasovius/misc/poisson_fft.h>

constexpr size_t order = 2;
using wendland_t       = vlasovius::kernels::wendland<1,order>;
using kernel_t         = vlasovius::kernels::tensorised_kernel<wendland_t>;
using interpolator_t   = vlasovius::interpolators::piecewise_interpolator<kernel_t>;

int main()
{
	constexpr double tikhonov_mu { 1e-12 };
	constexpr size_t min_per_box = 1000;

	double Lx = 4 * M_PI;
	double Ly = 4 * M_PI;
	size_t nx_poiss = 128;
	size_t ny_poiss = 128;
	size_t n_poiss = nx_poiss * ny_poiss;

	vlasovius::dim2::poisson<double> poiss(Lx, Ly, nx_poiss, ny_poiss);
    void *tmp = std::aligned_alloc( poiss.alignment, sizeof(double)*nx_poiss*ny_poiss );
	std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(tmp), std::free };

	wendland_t W;
	arma::rowvec sigma { 1, 0.5 };
	kernel_t   K ( W, sigma );
	kernel_t   Kx( W, arma::rowvec { sigma(0) } );

	size_t Nx = 64, Ny = 64, Nv = 128, Nw = 128;
	size_t N_total = Nx * Ny * Nv * Nw;
	std::cout << "Number of particles: " << N_total << ".\n";

	size_t num_threads = omp_get_max_threads();

	arma::mat xv;
	arma::vec f;

	double vmax = 6;
	xv.set_size( N_total, 4);
	f.resize( N_total );
	for(size_t i = 0; i < Nx; i++)
		for(size_t j = 0; j < Ny; j++)
			for(size_t k = 0; k < Nv; k++)
				for(size_t l = 0; l < Nw; l++)
				{
					double x = (i+0.5) * (Lx/Nx);
					double y = (j+0.5) * (Ly/Ny);
					double v = -vmax + k*(2*vmax/(Nv-1));
					double w = -vmax + l*(2*vmax/(Nw-1));

					size_t curr = l + Nw*(k + Nv*(j + Ny*i));

					xv(curr, 0 ) = x;
					xv(curr, 1 ) = y;
					xv(curr, 2 ) = v;
					xv(curr, 3 ) = w;

					// Linear Landau damping:
					f(curr) = 1/(2*M_PI) * std::exp(-0.5*(v*v+w*w))
							* (1 + 0.05*std::cos(0.5*x)*std::cos(0.5*y));
				}

	double t = 0, T = 30.25, dt = 1./8.;
	while ( t < T )
	{
		std::cout << "t = " << t << ". " << std::endl;
		vlasovius::misc::stopwatch clock;

		if ( t + dt > T ) dt = T - t;


		xv.col(0) += dt*xv.col(2);             // Move x particles
		xv.col(0) -= Lx * floor(xv.col(0) / Lx); // Set to periodic positions.
		xv.col(1) += dt*xv.col(3);             // Move x particles
		xv.col(1) -= Ly * floor(xv.col(1) / Ly); // Set to periodic positions.

		interpolator_t sfx { K, xv, f, min_per_box, tikhonov_mu, num_threads };

		#pragma omp parallel for
		for(size_t i = 0; i < n_poiss; i++)
		{
			// Compute new rho values.
		}

	}


	return 0;
}
