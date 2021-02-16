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

#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/kernels/periodised_kernel.h>
#include <vlasovius/interpolators/direct_interpolator.h>
#include <vlasovius/misc/periodic_poisson_1d.h>

namespace vlasovius
{

template <size_t k>
class xv_kernel
{
public:
	xv_kernel() = delete;
	xv_kernel( double sigma_x, double sigma_v, double L );

	arma::mat operator()( const arma::mat &xv1, const arma::mat &xv2 ) const;

	arma::mat eval_x( const arma::mat &xv1, const arma::mat &xv2 ) const;
	arma::mat eval_v( const arma::mat &xv1, const arma::mat &xv2 ) const;

private:
	using rbf_wendland = kernels::wendland<1,k>;
	using wendland     = kernels::rbf_kernel<rbf_wendland>;
	using xkernel_t    = kernels::periodised_kernel<wendland,1>;
	using vkernel_t    = wendland;

	xkernel_t Kx;
	vkernel_t Kv;
};


template <size_t k>
xv_kernel<k>::xv_kernel( double sigma_x, double sigma_v, double L ):
Kx { wendland { rbf_wendland {}, sigma_x }, L, (size_t) (L/sigma_v) },
Kv { rbf_wendland {}, sigma_v }
{}

template <size_t k>
arma::mat xv_kernel<k>::operator()( const arma::mat &xv1, const arma::mat &xv2 ) const
{
	return Kx( xv1.col(0), xv2.col(0) ) %
	       Kv( xv1.col(1), xv2.col(1) );
}

template <size_t k>
arma::mat xv_kernel<k>::eval_x( const arma::mat &xv1, const arma::mat &xv2 ) const
{
	return Kx( xv1.col(0), xv2.col(0) );
}

template <size_t k>
arma::mat xv_kernel<k>::eval_v( const arma::mat &xv1, const arma::mat &xv2 ) const
{
	return Kv( xv1.col(1), xv2.col(1) );
}

}

int main()
{
	constexpr size_t order = 4;
	using kernel_t        = vlasovius::xv_kernel<order>;
	using interpolator_t  = vlasovius::interpolators::direct_interpolator<kernel_t>;
	using poisson_t       = vlasovius::misc::poisson_gedoens::periodic_poisson_1d<8>;

	using wendland_t = vlasovius::kernels::wendland<1,order>;
	wendland_t W;

	double sigma_x = 4, sigma_v = 4, L = 4*3.14159265358979323846;
	kernel_t K( sigma_x, sigma_v, L );


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
	size_t Nx = 20, Nv = 72;
	xv.set_size( Nx*Nv,2 );
	f.resize( Nx*Nv );
	for ( size_t i = 0; i < Nx; ++i )
	for ( size_t j = 0; j < Nv; ++j )
	{
		double x = i * (L/Nx);
		double v = -5 + j*(10./(Nv-1));

		xv( j + Nv*i, 0 ) = x;
		xv( j + Nv*i, 1 ) = v;
		constexpr double alpha = 0.01;
		constexpr double K     = 0.5;
		f( j + Nv*i ) = 0.39894228040143267793994 * ( 1 + alpha*std::cos(K*x) ) * std::exp( -v*v/2 );
	}

	size_t plot_count = 0;
	double t = 0, T = 20, dt = 1./16.;
	while ( t < T )
	{
		std::cout << "t = " << t << ". "; std::cout.flush();
		for ( size_t stage = 0; stage < 4; ++stage )
		{
			arma::mat xv_stage = xv;
			for ( size_t s = 0;  s+1 <= stage; ++s )
				xv_stage += dt*c_rk4[stage][s]*k_xv[s];

			k_xv[ stage ].resize( xv.n_rows, xv.n_cols );
			k_xv[ stage ].col(0) = xv_stage.col(1);

			interpolator_t sfx( K, xv_stage, f, 1e-12 );
			arma::vec rho = arma::vec(rho_points.n_rows,arma::fill::ones) -
					        2 * W.integral() * sigma_v * sfx(rho_points);
			poisson.update_rho( rho );

			for ( size_t i = 0; i < xv_stage.n_rows; ++i )
				k_xv[stage](i,1) = -poisson.E( xv_stage(i,0) );

			if ( stage == 0 )
				std::cout << "E-max: " << norm( k_xv[stage].col(1), "inf" ) << "." << std::endl;
 		}

		for ( size_t s = 0; s < 4; ++s )
			xv += dt*d_rk4[s]*k_xv[s];
		t += dt;

		if ( t + dt > T ) dt = T - t;
	}
}
