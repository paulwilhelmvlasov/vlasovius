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
#include <vlasovius/misc/fields.hpp>

constexpr size_t order = 2;
using wendland_t       = vlasovius::kernels::wendland<1,order>;
using kernel_t         = vlasovius::kernels::tensorised_kernel<wendland_t>;
using interpolator_t   = vlasovius::interpolators::piecewise_interpolator<kernel_t>;

int main()
{
	// Note: I use the Poisson solver originally writen for the DerGerät solver.
	// It has a weirdish structure which does not really suit the vlasovius-solver,
	// however for testing reasons I write this version now in a compatible way
	// and will change it later for a potential release.

	double T = 30.25, dt = 1./8.;
	size_t Nt = T / dt + 1;

	constexpr double tikhonov_mu { 1e-12 };
	constexpr size_t min_per_box = 1000;

	double Lx = 4 * M_PI;
	double Ly = 4 * M_PI;
	size_t nx_poiss = 128;
	size_t ny_poiss = 128;
	size_t n_poiss = nx_poiss * ny_poiss;

	vlasovius::dim2::poisson<double> poiss(Lx, Ly, nx_poiss, ny_poiss);
	const size_t stride_x = 1;
	const size_t stride_y = stride_x*(nx_poiss + order - 1);
	const size_t stride_t = stride_y*(ny_poiss + order - 1);
	std::unique_ptr<double[]> coeffs { new double[ (Nt+1)*stride_t ] {} };

    void *tmp = std::aligned_alloc( poiss.alignment, sizeof(double)*nx_poiss*ny_poiss );
	std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(tmp), std::free };

	arma::mat rho_points(n_poiss, 2);
	constexpr size_t interpol_order_poiss = 4;
	double dx_poiss = Lx / nx_poiss;
	double dy_poiss = Ly / ny_poiss;
	for(size_t i = 0; i < nx_poiss; i++)
		for(size_t j = 0; j < ny_poiss; j++)
		{
			double x = i * dx_poiss;
			double y = j * dy_poiss;

			size_t curr = j + i * ny_poiss;
			rho_points(curr, 0) = x;
			rho_points(curr, 1) = y;
		}
	vlasovius::geometry::kd_tree rho_tree(rho_points);

	wendland_t W;
	arma::rowvec sigma { 1, 1, 1, 1 };
	kernel_t   K ( W, sigma );
	kernel_t   Kx( W, arma::rowvec { sigma(0) } );
	kernel_t   Ky( W, arma::rowvec { sigma(1) } );

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

					double c = 1.0 / (2.0 * M_PI);

					// Linear Landau damping:
					f(curr) = c * std::exp(-0.5*(v*v+w*w))
							* (1 + 0.05*std::cos(0.5*x)*std::cos(0.5*y));
				}

	vlasovius::dim2::config_t<double> conf;
	conf.Nx = nx_poiss;
	conf.Ny = ny_poiss;
	conf.u_min = -6;
	conf.v_min = -6;
	conf.u_max = 6;
	conf.v_max = 6;
	conf.x_min = 0;
	conf.y_min = 0;
	conf.x_max = Lx;
	conf.y_max = Ly;
	conf.dt = dt;
	conf.Nt = Nt;
	conf.Lx = Lx;
	conf.Ly = Ly;
	conf.Lx_inv = 1 / Lx;
	conf.Ly_inv = 1 / Ly;
	conf.dx = dx_poiss;
	conf.dy = dy_poiss;

	double t = 0;
	size_t n_step = 0;
	std::ofstream Emax_file( "Emax2d.txt" );
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

		// Compute new rho-values:
		arma::vec rho_tmp(n_poiss, arma::fill::ones);
		#pragma omp parallel
		{
			arma::vec my_rho(rho_tmp.size(),arma::fill::zeros);
			arma::vec coeff( 2*min_per_box );

			#pragma omp for schedule(dynamic)
			for ( size_t i = 0; i < sfx.cover.n_rows; ++i )
			{
				const auto &local = sfx.local_interpolants[i];
				double x_min = sfx.cover(i,0), y_min = sfx.cover(i,1),
					   v_min = sfx.cover(i,2), w_min = sfx.cover(i,3),
					   x_max = sfx.cover(i,4), y_max = sfx.cover(i,5),
					   v_max = sfx.cover(i,6), w_max = sfx.cover(i,7);

				size_t n = local.points().n_rows;
				const arma::mat &X = local.points();

				if ( n > coeff.size() ) coeff.resize(n);

				for ( size_t j = 0; j < n; ++j )
				{
					double v = X(j,2);
					double w = X(j,3);
					coeff(j) = local.coeffs()(j) * sigma(2) * sigma(3)
							 * ( W.integral((v-v_min)/sigma(2)) +
								 W.integral((v_max-v)/sigma(2)) )
						     * ( W.integral((w-w_min)/sigma(3)) +
							     W.integral((w_max-w)/sigma(3)) );
				}

				arma::rowvec box { x_min, y_min, x_max, y_max };
				arma::uvec idx = rho_tree.index_query(box);
				arma::vec tmpx = rho_points.col(0);
				arma::vec tmpy = rho_points.col(1);
				my_rho(idx) += Kx(tmpx(idx),X)*Ky(tmpy(idx),X)*coeff( arma::span(0,n-1) );
			}

			#pragma omp critical
			rho_tmp -= my_rho;
		}

		#pragma omp parallel for
		for(size_t i = 0; i < n_poiss; i++)
		{
			rho.get()[i] = rho_tmp(i);
		}

		poiss.solve( rho.get() ); // Now the potential phi is stored in rho.

		if ( n_step )
		{
			for ( size_t l = 0; l < stride_t; ++l )
				coeffs[ n_step*stride_t + l ] = coeffs[ (n_step-1)*stride_t + l ];
		}
		vlasovius::dim2::interpolate<double,interpol_order_poiss>( coeffs.get() + n_step*stride_t, rho.get(), conf );

        double Emax = 0;
        for ( size_t l = 0; l < n_poiss; ++l )
        {
            size_t i = l % nx_poiss;
            size_t j = l / nx_poiss;
            double x = i*dx_poiss;
            double y = j*dy_poiss;

            double Ex = -vlasovius::dim2::eval<double,interpol_order_poiss,1,0>( x, y, coeffs.get() + n_step*stride_t, conf );
            double Ey = -vlasovius::dim2::eval<double,interpol_order_poiss,0,1>( x, y, coeffs.get() + n_step*stride_t, conf );

            Emax = std::max( Emax, hypot(Ex,Ey) );
        }
        double elapsed = clock.elapsed();
        Emax_file << std::setw(15) << n_step*conf.dt << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl;
        std::cout << std::setw(15) << n_step*conf.dt << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl;

        std::cout << "This time step took: " << elapsed << "s." <<std::endl;


		t += dt;
		n_step++;
	}


	return 0;
}
