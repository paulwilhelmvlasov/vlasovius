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
#ifndef VLASOVIUS_INTERPOLATORS_DIRECTINTERPOLATOR_H
#define VLASOVIUS_INTERPOLATORS_DIRECTINTERPOLATOR_H

#include <vector>

namespace vlasovius
{
	typedef point = arma::vec; 

	namespace interpolators
	{
		template <typename kernel>
		class directInterpolator
		{
		public:
			directInterpolator(const std::vector<point>& X, const std::vector<double>& f, double sigma = 1.0);
			
		public:
			double operator(const point& x) const;

		private:
			arma::mat computeRKHSMatrix(const std::vector<point>& X);
			void computeCoeffVec(const arma::mat& rkhsMat, const std::vector<double>& f);

		private:
			long N;

			arma::vec coeff;
			const std::vector<point> interPolPoints; 
		};
	}

}

#include <vlasovius/interpolators/direct_interpolators.cpp>
#endif
