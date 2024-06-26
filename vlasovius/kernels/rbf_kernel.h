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
#ifndef VLASOVIUS_KERNELS_RBF_KERNEL_H
#define VLASOVIUS_KERNELS_RBF_KERNEL_H

#include <armadillo>
#include <vlasovius/misc/simd.h>

namespace vlasovius
{

namespace kernels
{

template <typename rbf_function>
class rbf_kernel
{
public:
	using simd_t = typename rbf_function::simd_type;

	rbf_kernel( rbf_function p_F = rbf_function {}, double sigma = 1 );

	rbf_kernel( const rbf_kernel&  ) = default;
	rbf_kernel(       rbf_kernel&& ) = default;
	rbf_kernel& operator=( const rbf_kernel&  ) = default;
	rbf_kernel& operator=(       rbf_kernel&& ) = default;

	arma::mat operator()( const arma::mat &X, const arma::mat &Y ) const;

	void eval( size_t dim, size_t n, size_t m,
			         double *K, size_t ldK,
	           const double *X, size_t ldX,
	           const double *Y, size_t ldY, size_t num_threads = 1 ) const;

	void mul ( size_t dim, size_t n, size_t m, size_t nrhs,
			         double *R, size_t ldK,
	           const double *X, size_t ldX,
	           const double *Y, size_t ldY,
			   const double *C, size_t ldC, size_t num_threads = 1 ) const;

private:

	void eval_column( size_t dim, size_t n, size_t j,
			                double *K, size_t ldK,
	                  const double *X, size_t ldX,
	                  const double *Y, size_t ldY ) const;

	void micro_kernel( size_t dim, simd_t mat[ simd_t::size() ],
			           const double *X, size_t ldX,
	                   const double *Y, size_t ldY ) const;

	rbf_function F   {};
	double inv_sigma {1};
};

}

}

#include <vlasovius/kernels/rbf_kernel.tpp>
#endif
