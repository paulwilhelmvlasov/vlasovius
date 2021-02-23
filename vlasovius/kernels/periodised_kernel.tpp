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

template <typename base_kernel>
periodised_kernel<base_kernel,1>::
periodised_kernel( base_kernel p_K, double p_L, size_t p_num_images ):
K { std::move( p_K ) }, L { p_L }, num_images { p_num_images }
{
	if ( L < 0 )
	{
		throw std::logic_error { "vlasovius::kernels::periodised_kernel<base_kernel,1>: "
			                     "Specified negative period length." };
	}

	if ( ! std::isnormal(L) )
	{
		throw std::logic_error { "vlasovius::kernels::periodised_kernel<base_kernel,1>: "
			                     "Specified anormal period length." };
	}
}

template <typename base_kernel>
arma::mat periodised_kernel<base_kernel,1>::operator()( arma::vec X, arma::vec Y ) const
{
	double Linv = 1/L;
	X -= L*floor(X*Linv);
	Y -= L*floor(Y*Linv);

	arma::mat result = K(X,Y);
	for ( size_t n = 1; n <= num_images; ++n )
	{
		result += K(X, Y + n*L );
		result += K(X, Y - n*L );
	}

	return result;
}

}

}
