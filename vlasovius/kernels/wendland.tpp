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

namespace wendland_impl
{

void compute_coefficients( size_t dim, size_t k, double *result );

}

template <size_t dim, size_t k>
wendland<dim,k>::wendland()
{
	wendland_impl::compute_coefficients( dim, k, c );
}

template <size_t dim, size_t k>
double wendland<dim,k>::operator()( double r ) const noexcept
{
	constexpr size_t N { (dim/2) + 3*k + 2 };

	r = std::abs(r);
	if ( r >= 1 ) return 0;

	// Clenshaw's algorithm for evaluating Chebyshev expansions.
	double f      { 4*r - 2 };
	double z_prev { c[0] };
	double z      { std::fma(f,c[0],c[1]) };
	for ( size_t i = 2; i < N-1; ++i )
	{
		double tmp = std::fma(f,z,c[i]) - z_prev;
		z_prev = z;
		z      = tmp;
	}

	f = 0.5*f;
	return std::fma(f,z,c[N-1]) - z_prev;
}

#if defined(HAVE_AVX_INSTRUCTIONS) && defined(HAVE_FMA_INSTRUCTIONS)

template <size_t dim, size_t k>
arma::vec wendland<dim,k>::operator()( arma::vec rvec ) const
{
	constexpr size_t N { (dim/2) + 3*k + 2 };

	__m256d cc[N];
	for ( size_t i = 0; i < N; ++i )
		cc[i] = _mm256_broadcast_sd( &c[i] );

	__m256d sign_mask  = _mm256_set1_pd(-0.0);
	__m256d ones       = _mm256_set1_pd( 1.0);

	size_t num_chunks = rvec.size()/4;
	#pragma omp parallel for schedule(static)
	for ( size_t chunk = 0; chunk < num_chunks; ++chunk )
	{
		__m256d r = _mm256_loadu_pd( rvec.memptr() + 4*chunk );
		r = _mm256_andnot_pd( sign_mask, r ); // r = abs(r);

		__m256d threshold = _mm256_cmp_pd( r, ones, _CMP_LT_OQ );
		if ( _mm256_testz_pd(threshold,threshold) )
		{
			// if all r >= 1 store zero and continue.
			_mm256_storeu_pd( rvec.memptr() + 4*chunk,  _mm256_setzero_pd() );
			continue;
		}


		// Clenshaw algorithm.
		__m256d fhalf  = _mm256_add_pd(r,r);
		        fhalf  = _mm256_sub_pd(fhalf,ones);			// f_half = 2*r - 1;
		__m256d f      = _mm256_add_pd(fhalf,fhalf);		// f      = 4*r - 2;
		__m256d z_prev = cc[0];
		__m256d z      = _mm256_fmadd_pd(f,cc[0],cc[1]);	// z = f*c[0] - c[1];
		for ( size_t i = 2; i < N-1; ++i )
		{
			__m256d tmp = _mm256_fmadd_pd(f,z,cc[i]);
			tmp = _mm256_sub_pd( tmp, z_prev );  // tmp = f*z + c[i] - z_prev;
			z_prev = z;
			z      = tmp;
		}
		z = _mm256_fmadd_pd(fhalf,z,cc[N-1]);
		z = _mm256_sub_pd( z, z_prev );
		z = _mm256_and_pd( z, threshold ); // Set zero if r >= 1.
		_mm256_storeu_pd( rvec.memptr() + 4*chunk,  z );
	}

	// Compute the remaining entries in scalar mode.
	for ( size_t i = 4*num_chunks; i < rvec.size(); ++i )
		rvec[i] = (*this)(rvec[i]);

	return rvec;
}

#else

template <size_t dim, size_t k>
arma::vec wendland<dim,k>::operator()( arma::vec r ) const
{
	#pragma omp parallel for schedule(static)
	for ( size_t i = 0; i < r.size(); ++i )
		r[i] = (*this)(r[i]);
	return r;
}

#endif

}

}
