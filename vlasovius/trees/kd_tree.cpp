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

#include "vlasovius/trees/kd_tree.h"

namespace vlasovius
{

	namespace trees
	{

		template<arma::uword dim> bool compVec
		(const arma::vec& first, const arma::vec& second)
		{
			return (first(dim) < second(dim));
		}

		kd_tree::kd_tree(arma::mat& points, size_t minPerBox, size_t maxPerBox)
		{
			// Init first node:
			nodes.push_back();
			nodes[0].indexFirstElem = 0;
			nodes[0].indexLastElem  = points.n_rows;

			nodes[0].box.center     = points.row(0);
			nodes[0].box.sidelegnth = points.row(0);

			arma::uword dim = points.n_cols;

			// Compute its bounding box:
			for(arma::uword i = 0; i < dim; i++){
				double min 		  = points.col(i).min();
				double max 		  = points.col(i).max();
				double sidelength = (max - min) / 2.0;

				nodes[0].box.sidelength(i) = sidelength;
				nodes[0].box.center(i)     = max - sidelength;
			}

			buildTree(points, 0, minPerBox, maxPerBox);
		}


		void kd_tree::buildTree(arma::mat& points, size_t currentNodeIndex, size_t minPerBox, size_t maxPerBox)
		{
			// Use the standard splitting rule for kd-trees, i.e.:
			// The split-dimension is the one with maximum spread
			// of points and the splitting value is the median of
			// the respective coordinates of the points in the
			// current box.

			size_t numIndicesInBox = nodes[currentNodeIndex].point_indices.size();
			if(numIndicesInBox > maxPerBox && (numIndicesInBox / 2) >= minPerBox){
				// In this case split the node:

				// Init new nodes:
				nodes.push_back(node());
				nodes.push_back(node());

				size_t firstChild  = nodes.size() - 2;
				size_t secondChild = nodes.size() - 1;

				nodes[currentNodeIndex].firstChild  = firstChild;
				nodes[currentNodeIndex].secondChild = secondChild;

				nodes[firstChild].parent  = currentNodeIndex;
				nodes[secondChild].parent = currentNodeIndex;

				// Compute splitting dimension:
				size_t dimSplit = splittingDimension(currentNodeIndex);

				// Split along the dimension at the correct value:
				split(points, currentNodeIndex, dimSplit);

				// Compute boxes for children:
				nodes[firstChild].box  = nodes[currentNodeIndex].box;
				nodes[sceondChild].box = nodes[currentNodeIndex].box;

				double lowerBorder = points.row(nodes[firstChild].indexFirstElem)(dimSplit);
				double splitValue  = points.row(nodes[firstChild].indexLastElem)(dimSplit);
				double upperBorder = points.row(nodes[secondChild].indexLastElem)(dimSplit);

				double firstSideLength  = (splitValue - lowerBorder) / 2.0;
				double secondSideLength = (upperBorder - splitValue) / 2.0;

				nodes[firstChild].box.sidelength(dimSplit)  = firstSideLength;
				nodes[secondChild].box.sidelength(dimSplit) = secondSideLength;

				nodes[firstChild].box.center(dimSplit) -= firstSideLegnth;
				nodes[secondChild].box.center(dimSplit) -= secondSideLegnth;

				// Start recursion for children:
				buildTree(points, firstChild, minPerBox, maxPerBox);
				buildTree(points, secondChild, minPerBox, maxPerBox);
			}
		}


		size_t kd_tree::splittingDimension(size_t currentNodeIndex)
		{
			// We use the maximum sidelength of the current box as an
			// approximation to the maximum spread. While this might
			// not be 100% accurate in some cases, it will certainly
			// give a good approximation in most cases.

			if(dim == 1){
				return 0;
			} else{
				size_t dimSplit = 0;
				double maxLength = nodes[currentNodeIndex].box.sidelength(0);

				for(size_t i = 1; i < dim; i++)
				{
					double currLength = nodes[currentNodeIndex].box.sidelength(i);
					if(maxLength < currLength)
					{
						dimSplit = i;
						maxLength = currLength;
					}
				}

				return dimSplit;
			}
		}

		void kd_tree::split(arma::mat& points, size_t currentNodeIndex, size_t dimSplit)
		{
			arma::uword first = nodes[currentNodeIndex].indexFirstElem;
			arma::uword last  = nodes[currentNodeIndex].indexLastElem + 1;
			arma::uword nth = (last - first) / 2;
			std::nth_element(row_iter(points, first),
					row_iter(points, nth),
					row_iter(points, last),
					compVec<dimSplit>);

			nodes[nodes[currentNodeIndex].firstChild].indexFirstElem = first;
			nodes[nodes[currentNodeIndex].firstChild].indexLastElem = nth;

			nodes[nodes[currentNodeIndex].secondChild].indexFirstElem = nth + 1;
			nodes[nodes[currentNodeIndex].secondChild].indexLastElem = last - 1;
		}

		bool bounding_box::contains(const arma::vec& p)
		{
			return abs(center - p) <= sidelength;
		}

		bool node::isLeaf()
		{
			return (firstChild < 0);
			// If children are not set, i.e. == -1,
			// the node is a leaf.
		}

		node kd_tree::getNode(size_t i) const
		{
			return nodes.at(i);
		}

		int kd_tree::whichBoxContains(const arma::vec& p) const
		{
			if(nodes[0].box.contains(p)){
				// Trace the tree for the leaf-node containing
				// the point:
				int index = 0;
				do
				{
					int firstChild = nodes[index].firstChild;
					int secondChild = nodes[index].secondChild;
					if( nodes[firstChild].box.contains(p) ){
						index = firstChild;
					} else {
						index = secondChild;
					}
				} while( ! nodes[index].isLeaf() );

				return index;
			}else{
				// Passed point is not inside the tree:
				return -1;
			}
		}

		int whichBoxContains(size_t i) const
		{
			if(i <= nodes[0].indexLastElem )
			{
				// Trace the tree for the leaf-node containing
				// the point:
				int index = 0;
				do
				{
					int firstChild = nodes[index].firstChild;
					int secondChild = nodes[index].secondChild;
					if( i <= nodes[firstChild].indexLastElem ){
						index = firstChild;
					} else {
						index = secondChild;
					}
				} while( ! nodes[index].isLeaf() );

				return index;
			}else{
				// Passed point is not inside the tree:
				return -1;
			}

		}

	}
}
