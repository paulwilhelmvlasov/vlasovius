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

#include "vlasovius/interpolators/pou_interpolator.h"

template <size_t k>
vlasovius::interpolators::pou_inducing_kernel<k>::pou_inducing_kernel( const arma::rowvec& sigma)
{
	size_t d = sigma.size();
	dim_kernels.reserve(d);

	for(size_t i = 0; i < d; i++)
	{
		dim_kernels.push_back(wendland{rbf_wendland {}, sigma(i) });
	}
}

template <size_t k>
arma::mat vlasovius::interpolators::pou_inducing_kernel<k>::operator()( const arma::mat &xv1, const arma::mat &xv2 ) const
{
	arma::mat m = dim_kernels[0]( xv1.col(0), xv2.col(0) ) %
		       dim_kernels[1]( xv1.col(1), xv2.col(1) );

	for(size_t i = 2; i < dim_kernels.size(); i++)
	{
		m = m % dim_kernels[i]( xv1.col(i), xv2.col(i) );
	}

	return m;
}

template <typename kernel>
vlasovius::interpolators::pou_interpolator<kernel>::pou_interpolator( kernel K, arma::mat X,
		arma::vec b, double tikhonov_mu, size_t min_per_box, size_t max_per_box,
		double enlargement_factor)
: K(K), tree(vlasovius::trees::kd_tree(X, b, min_per_box, max_per_box))
{
	if(enlargement_factor < 1.0) {
		throw std::runtime_error("Enlargement factor must be greater or equal 1.");
	}

	sub_sfx.reserve(tree.getNumberLeafs());
	weight_fcts.reserve(tree.getNumberLeafs());

	construct_sub_sfx(X, b, enlargement_factor, tikhonov_mu);
}

template <typename kernel>
void vlasovius::interpolators::pou_interpolator<kernel>::construct_sub_sfx(arma::mat X,
		arma::vec b, double enlargement_factor, double tikhonov_mu)
{
	// Get the leaf-indices:
	// (Note that usually there are approximately twice as many nodes as leafs
	// so just running through all nodes and checking on leaf-status might have
	// the optimal run-time.)
	size_t n_nodes = tree.get_number_nodes();
	size_t n_leafs = tree.getNumberLeafs();

	std::vector<size_t> indices_leafs;
	indices_leafs.reserve(n_leafs);

	for(size_t i = 0; i < n_nodes; i++){
		if(tree.getNode(i).isLeaf()){
			indices_leafs.push_back(i);
		}
	}

	// Get the point-sets for each sub-sfx and compute the bounding-box. Note that
	// I have to enlarge the boxes of each leaf such that they slightly overlap.
	// To compute the enlarged boxes, use the tree-structure.
	std::vector<std::deque<arma::uword>> indices_points(n_leafs);
	sub_domains = std::vector<vlasovius::trees::bounding_box>(n_leafs);

	for(size_t i = 0; i < n_leafs; i++) // Can I pragma parallel for this?
	{
		size_t index_node = indices_leafs[i];
		vlasovius::trees::node leaf_nd = tree.getNode(index_node);

		//Init local point deque:
		indices_points[i] = std::deque<arma::uword>(leaf_nd.indexLastElem - leaf_nd.indexFirstElem);

		// Fill index-list with current inhabitants:
		std::iota(indices_points[i].begin() + leaf_nd.indexFirstElem,
				indices_points[i].begin() + leaf_nd.indexLastElem,
				leaf_nd.indexFirstElem);

		// Compute new bounding-box:
		sub_domains[i] = leaf_nd.box;
		sub_domains[i].sidelength *= enlargement_factor;

		// Find now all points intersecting the new bounding-box:
		// Therefore first find the parent-node which completely contains the sub-domain
		// and check child-points on intersection.
		int index_curr_parent = leaf_nd.parent;

		// While root (== 0) is not reached and the sub-domain is not a subset of the current
		// bounding box trace the tree to the top.
		while(index_curr_parent != 0 || !subset(sub_domains[i], tree.getNode(index_curr_parent).box))
		{
			index_curr_parent = tree.getNode(index_curr_parent).parent;
		}

		// Now check the contained points on intersection. Ignore the already known points (i.e.
		// the ones contained in the original leaf):
		vlasovius::trees::node parent_nd = tree.getNode(index_curr_parent);
		for(arma::uword j = parent_nd.indexFirstElem;
				(j < parent_nd.indexLastElem)
			&& !(leaf_nd.indexFirstElem <= j && j <= leaf_nd.indexLastElem);
				j++ )
		{
			if(sub_domains[i].contains(X.row(j)))
			{
				indices_points[i].push_back(j);
			}
		}

		// Finally compute the direct_interpolator's:
		arma::uword N_sub = indices_points[i].size();
		arma::mat sub_pts(N_sub, X.n_cols);
		arma::vec sub_rhs(N_sub);

		#pragma omp parallel for
		for(size_t j = 0; j < indices_points[i].size(); j++)
		{
			arma::uword curr = indices_points[i][j];
			sub_pts.row(j) = X.row(curr);
			sub_rhs(j) = b(curr);
		}

		sub_sfx.push_back(direct_interpolator<kernel>(K, sub_pts, sub_rhs, tikhonov_mu));
		weight_fcts.push_back(pou_inducing_kernel<4>(sub_domains[i].sidelength));
	}
}
