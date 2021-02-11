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
 
#ifndef VLASOVIUS_TREES_QUADTREE_H
#define VLASOVIUS_TREES_QUADTREE_H

#include <vector>
#include <deque>
#include <armadillo>

namespace vlasovius
{
	namespace trees
	{
		struct bounding_box_2d
		{
			bool pointInsideAABB(const arma::vec2& x, 
				double tol = 1e-16);

			double radius;
			arma::vec2 center;
		};

		struct node
		{
			size_t parent   = 0;

			size_t topLeft  = 0;
			size_t topRight = 0;
			size_t botRight = 0;
			size_t botLeft  = 0;

			std::vector<size_t> childIndices = {};

			bounding_box_2d box;
		};

		class quadtree
		{
		public:
			quadtree(arma::mat& points, size_t minElemPerBox, size_t maxElemPerBox);


		private:
			void buildTree(arma::mat& points, 
				size_t minElemPerBox, size_t maxElemPerBox, 
				node& currNode, size_t currIndex);

		private:
			std::deque<node> nodes;

			arma::mat points;
		};
	}
 }

#endif