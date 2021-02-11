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

namespace vlasovius
{

namespace kernels
{

template <typename rbf_function>
rbf_kernel<rbf_function>::rbf_kernel( rbf_function p_F, double sigma ):
F { p_F }, inv_sigma { 1./sigma }
{}

template <typename rbf_function>
arma::mat rbf_kernel<rbf_function>::operator()( const arma::mat &X,
		                                        const arma::mat &Y ) const
{
	if ( X.n_cols != Y.n_cols )
	{
		throw std::logic_error { "vlasovius::kernels::rbf_kernel::operator(): "
			                     "X and Y contain points of differing dimension." };
	}

	size_t dim { X.n_cols };
	size_t N { X.n_rows }, M { Y.n_rows };

	arma::vec ones( dim, arma::fill::ones );
	arma::vec vx = (X%X) * ones;
	arma::vec vy = (Y%Y) * ones;

	arma::mat result = -2*X*Y.t();
	for ( size_t i = 0; i < N; ++i )
		result.row(i) += vy.t();
	for ( size_t j = 0; j < M; ++j )
		result.col(j) += vx;

	result = inv_sigma*sqrt(abs(result));


	// Interpretation of result matrix as vector.
	arma::vec result_vec( result.memptr(), N*M, false, true );

	// Make use of vectorised RBF-kernel.
	result_vec = F( std::move(result_vec) );

	return result;
}

}

}
