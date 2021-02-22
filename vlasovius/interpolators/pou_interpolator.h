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
#ifndef VLASOVIUS_INTERPOLATORS_POU_INTERPOLATOR_H
#define VLASOVIUS_INTERPOLATORS_POU_INTERPOLATOR_H

#include <vector>
#include <armadillo>

#include <vlasovius/interpolators/direct_interpolator.h>
#include <vlasovius/kernels/wendland.h>
#include <vlasovius/kernels/rbf_kernel.h>
#include <vlasovius/trees/kd_tree.h>

namespace vlasovius
{
	namespace interpolators
	{

		template <size_t k>
		class pou_inducing_kernel
		{
		public:
			pou_inducing_kernel() = delete;
			pou_inducing_kernel( const arma::rowvec& sigma);

			arma::mat operator()( const arma::mat &xv1, const arma::mat &xv2 ) const;

		private:
			using rbf_wendland = vlasovius::kernels::wendland<1,k>;
			using wendland     = vlasovius::kernels::rbf_kernel<rbf_wendland>;

			std::vector<wendland> dim_kernels;
		};


		template <typename kernel>
		class pou_interpolator
		{
		public:
			pou_interpolator( kernel K, arma::mat X, arma::vec b,
					double tikhonov_mu = 0, size_t min_per_box = 100,
					size_t max_per_box = 200,
					double enlargement_factor = 1.5);

			arma::vec operator()( const arma::mat &Y ) const;

		private:
			void construct_sub_sfx(arma::mat X, arma::vec b,
					double enlargement_factor, double tikhonov_mu);

		private:
			kernel    K;
			vlasovius::trees::kd_tree tree;

			std::vector<direct_interpolator<kernel>> sub_sfx;
			std::vector<pou_inducing_kernel<4>> weight_fcts;
			std::vector<vlasovius::trees::bounding_box> sub_domains;
		};
	}

}

#include <vlasovius/interpolators/pou_interpolator.tpp>
#endif
