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

		bool subset(const bounding_box& first, const bounding_box& second)
		{
			#ifndef NDEBUG
			if ( first.center.size() != second.center.size() )
			{
				throw std::runtime_error { "vlasovius::trees::intersects: "
										   "Comparison of boxes of differing dimension." };
			}
			#endif

			const arma::uword dim { second.center.size() };
			for( arma::uword d = 0; d < dim; ++d )
			{
				double dist = std::abs(first.center(d) - second.center(d)) -
									  (first.sidelength(d) + second.sidelength(d));
				if ( dist > 0 )
				{
					return false;
				}
			}

			return true;
		}

		bool intersect(const bounding_box& first, const bounding_box& second)
		{
			size_t d = second.center.size();

			for(size_t i = 0; i < d; i++)
			{
				double dist = std::abs(first.center(i) - second.center(i));
				if( dist <= first.sidelength(i) || dist <= second.sidelength(i) )
				{
					return true;
				}
			}

			return false;
		}

		kd_tree::kd_tree(arma::mat& points, arma::vec& rhs, size_t minPerBox, size_t maxPerBox)
		{
			vlasovius::misc::stopwatch clock;
			// Set dimension of points:
			dim = points.n_cols;

			// Init first node:
			nodes.push_back(node());
			nodes[0].indexFirstElem = 0;
			nodes[0].indexLastElem  = points.n_rows;

			nodes[0].box = computeBox(points);

			n_leafs = 1;

			std::vector<arma::uword> sortedIndices(points.n_rows);
			std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

			buildTree(sortedIndices, points, 0, minPerBox, maxPerBox);

			sortPoints(sortedIndices, points, rhs);

			indices_leafs.reserve(n_leafs);

			for(size_t i = 0; i < nodes.size(); i++){
				if(nodes[i].isLeaf()){
					indices_leafs.push_back(i);
					node_index_leaf_index.insert({i, indices_leafs.size() - 1});
				}
			}

			double elapsed { clock.elapsed() };
			std::cout << "Time for computing kd-tree: " << elapsed << ".\n";
		}

		bounding_box computeBox(arma::mat& points)
		{
			bounding_box box;
			size_t dim = points.n_cols;
			box.center 		= points.row(0);
			box.sidelength  = points.row(0);

			for(arma::uword i = 0; i < dim; i++){
				double min 		  = points.col(i).min();
				double max 		  = points.col(i).max();
				double sidelength = (max - min) / 2.0;

				box.sidelength(i) = sidelength;
				box.center(i)     = max - sidelength;
			}

			return box;
		}

		void kd_tree::sortPoints(std::vector<arma::uword>& sortedIndices,
							arma::mat& points, arma::vec& rhs)
		{
			arma::mat copy_points(points);
			arma::vec copy_rhs(rhs);
			#pragma omp parallel for
			for(size_t i = 0; i < sortedIndices.size(); i++){
				arma::uword new_index = sortedIndices[i];
				points.row(i) = copy_points.row(new_index);
				rhs(i) 		  = copy_rhs(new_index);
			}
		}


		void kd_tree::buildTree(std::vector<arma::uword>& sortedIndices,
				arma::mat& points, size_t currentNodeIndex,
				size_t minPerBox, size_t maxPerBox)
		{
			// Use the standard splitting rule for kd-trees, i.e.:
			// The split-dimension is the one with maximum spread
			// of points and the splitting value is the median of
			// the respective coordinates of the points in the
			// current box.
			size_t numIndicesInBox = nodes[currentNodeIndex].indexLastElem
					- nodes[currentNodeIndex].indexFirstElem;
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
				split(sortedIndices, points, currentNodeIndex, dimSplit);

				// Compute boxes for children:
				nodes[firstChild].box  = nodes[currentNodeIndex].box;
				nodes[secondChild].box = nodes[currentNodeIndex].box;

				double lowerBorder = points.row(nodes[firstChild].indexFirstElem)(dimSplit);
				double splitValue  = points.row(nodes[firstChild].indexLastElem - 1)(dimSplit);
				double upperBorder = points.row(nodes[secondChild].indexLastElem - 1)(dimSplit);

				double firstSideLength  = (splitValue - lowerBorder) / 2.0;
				double secondSideLength = (upperBorder - splitValue) / 2.0;

				nodes[firstChild].box.sidelength(dimSplit)  = firstSideLength;
				nodes[secondChild].box.sidelength(dimSplit) = secondSideLength;

				nodes[firstChild].box.center(dimSplit) -= firstSideLength;
				nodes[secondChild].box.center(dimSplit) -= secondSideLength;

				// Start recursion for children:
				n_leafs++; // 1 leaf-node split into 2 leafs => Increment leaf-count.
				buildTree(sortedIndices, points, firstChild, minPerBox, maxPerBox);
				buildTree(sortedIndices, points, secondChild, minPerBox, maxPerBox);
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

		void kd_tree::split(std::vector<arma::uword>& sortedIndices,
				arma::mat& points, size_t currentNodeIndex, size_t dimSplit)
		{
			arma::uword first = nodes[currentNodeIndex].indexFirstElem;
			arma::uword last  = nodes[currentNodeIndex].indexLastElem;
			arma::uword nth = first + (last - first) / 2;

			auto comp = [&](arma::uword i, arma::uword j)->bool {
				return points(i, dimSplit) < points(j, dimSplit);
			};

			std::nth_element(sortedIndices.begin() + first,
					sortedIndices.begin() + nth,
					sortedIndices.begin() + last,
					comp);

			nodes[nodes[currentNodeIndex].firstChild].indexFirstElem = first;
			nodes[nodes[currentNodeIndex].firstChild].indexLastElem = nth;

			nodes[nodes[currentNodeIndex].secondChild].indexFirstElem = nth;
			nodes[nodes[currentNodeIndex].secondChild].indexLastElem = last;
		}

		bool bounding_box::contains(const arma::rowvec& p) const
		{
			size_t dim = p.n_cols;
			for(size_t i = 0; i < dim; i++){
				double dist = std::abs(center(i) - p(i));
				if(dist > sidelength(i))
				{
					return false;
				}
			}

			return true;
		}

		bool node::isLeaf() const
		{
			return (firstChild < 0);
			// If children are not set, i.e. == -1,
			// the node is a leaf.
		}

		size_t kd_tree::getNumberLeafs() const
		{
			return n_leafs;
		}

		size_t kd_tree::get_number_nodes() const
		{
			return nodes.size();
		}

		node kd_tree::getNode(size_t i) const
		{
			return nodes.at(i);
		}

		int kd_tree::whichLeafContains(const arma::rowvec& p) const
		{
			if(nodes[0].box.contains(p)){
				// Trace the tree for the leaf-node containing
				// the point:
				int index = 0;
				do
				{
					int firstChild = nodes[size_t(index)].firstChild;
					int secondChild = nodes[size_t(index)].secondChild;
					if( nodes[size_t(firstChild)].box.contains(p) ){
						index = firstChild;
					} else {
						index = secondChild;
					}
				} while( ! nodes[size_t(index)].isLeaf() );

				return node_index_leaf_index.at(size_t(index));
				// Returns the index of the leaf (in the leaf-list).
			}else{
				// Passed point is not inside the tree:
				return -1;
			}
		}

		int kd_tree::whichLeafContains(size_t i) const
		{
			if(i <= nodes[0].indexLastElem )
			{
				// Trace the tree for the leaf-node containing
				// the point:
				int index = 0;
				do
				{
					int firstChild = nodes[size_t(index)].firstChild;
					int secondChild = nodes[size_t(index)].secondChild;
					if( i <= nodes[size_t(firstChild)].indexLastElem ){
						index = firstChild;
					} else {
						index = secondChild;
					}
				} while( ! nodes[size_t(index)].isLeaf() );

				return node_index_leaf_index.at(size_t(index));
				// Returns the index of the leaf (in the leaf-list).
			}else{
				// Passed point is not inside the tree:
				return -1;
			}

		}

		std::vector<size_t> kd_tree::get_indices_leafs() const
		{
			return indices_leafs;
		}

	}
}
