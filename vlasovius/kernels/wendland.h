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
#ifndef VLASOVIUS_KERNELS_WENDLAND_H
#define VLASOVIUS_KERNELS_WENDLAND_H

#include <armadillo>
#include <vlasovius/misc/simd.h>

namespace vlasovius
{

namespace kernels
{

template <size_t dim, size_t k, typename simd_t = ::vlasovius::misc::simd<double> >
class wendland
{
public:
	using simd_type = simd_t;
	static constexpr size_t vecsize = simd_t::size();

	wendland();

	// Copies sometimes could make sense, when passing a kernel by value and
	// not ‘by type’ or by reference.
	wendland( const wendland&  ) noexcept = default;
	wendland(       wendland&& ) noexcept = default;

	// Assignment doesn't do anything
	wendland& operator=( const wendland&  ) noexcept { return *this; }
	wendland& operator=(       wendland&& ) noexcept { return *this; }


	// The heart of the matter: evaluation.
	double    operator()( double r    ) const noexcept;
	simd_t    operator()( simd_t v    ) const noexcept;
	arma::vec operator()( arma::vec r ) const;

	void eval( double *r, size_t n ) const noexcept;
	void eval( simd_t  r[vecsize] )  const noexcept;

	double integral( double r = 1 ) const noexcept;

private:
	double  c    [ (dim/2) + 3*k + 2 ]; // Chebyshev-Coefficients of Wendland function.
	double  c_int[ (dim/2) + 3*k + 3 ]; // Chebyshev-Coefficients of its integral.
};

}

}

#include <vlasovius/kernels/wendland.tpp>
#endif
