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
#ifndef VLASOVIUS_INTERPOLATORS_DIRECT_INTERPOLATOR_H
#define VLASOVIUS_INTERPOLATORS_DIRECT_INTERPOLATOR_H

#include <vector>
#include <armadillo>

namespace vlasovius
{

namespace interpolators
{

template <typename kernel>
class direct_interpolator
{
public:
	direct_interpolator() = default;
	direct_interpolator( const direct_interpolator&  )  = default;
	direct_interpolator(       direct_interpolator&&  ) = default;

	direct_interpolator& operator=( const direct_interpolator&  ) = default;
	direct_interpolator& operator=(       direct_interpolator&& ) = default;

	direct_interpolator( kernel K, arma::mat X, arma::mat b,
			             double tikhonov_mu = 0, size_t threads = 1 );

	arma::mat operator()( const arma::mat &Y, size_t threads = 1 ) const;
	const arma::mat& coeffs() const noexcept { return coeff; }
	const arma::mat& points() const noexcept { return X; }

private:
	kernel    K;
	arma::mat X;
	arma::mat coeff;
};

}

}

#include <vlasovius/interpolators/direct_interpolator.tpp>
#endif

