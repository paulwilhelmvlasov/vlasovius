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

#ifndef VLASOVIUS_KERNELS_PERIODISED_KERNEL_H
#define VLASOVIUS_KERNELS_PERIODISED_KERNEL_H

#include <cmath>
#include <armadillo>

namespace vlasovius
{

namespace kernels
{

/*!
 * \brief Turns a kernel on the whole space into a periodic one on a box.
 *
 * Let K(x,y) denote a kernel on the dim-dimensional whole space. Then this
 * class generates a kernel that is periodic on the domain
 * [0,L0] x [0,L1] x [0,L2] x ... x [0,Ldim-1] by adding mirror images.
 */
template <typename base_kernel, size_t dim>
class periodised_kernel;



template <typename base_kernel>
class periodised_kernel<base_kernel,1>
{
public:
	periodised_kernel( base_kernel p_K, double p_L, size_t p_num_images );
	periodised_kernel( const periodised_kernel&  ) = default;
	periodised_kernel(       periodised_kernel&& ) = default;

	periodised_kernel() = delete;
	periodised_kernel& operator=( const periodised_kernel&  ) = delete;
	periodised_kernel& operator=(       periodised_kernel&& ) = delete;

	arma::mat operator()( arma::vec X, arma::vec Y ) const;

private:
	base_kernel K;
	double      L;          // Box in which we want to be periodic.
	size_t      num_images; // Number of images we need to add in each direction.
};

}

}

#include <vlasovius/kernels/periodised_kernel.tpp>
#endif
