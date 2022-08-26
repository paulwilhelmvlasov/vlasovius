/*
 * Copyright (C) 2022 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of Der Gerät, a solver for the Vlasov–Poisson equation.
 *
 * Der Gerät is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * Der Gerät is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Der Gerät; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */

#ifndef VLASOVIUS_POISSON_POISSONFFT_HPP
#define VLASOVIUS_POISSON_POISSONFFT_HPP

#include <fftw3.h>

namespace vlasovius
{

namespace dim1
{
	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
        const size_t alignment { 64 };

		poisson() = delete;
		poisson( const poisson  &rhs ) = delete;
		poisson(       poisson &&rhs ) = delete;
		poisson& operator=( const poisson  &rhs ) = delete;
		poisson& operator=(       poisson &&rhs ) = delete;
		poisson(double Lx, size_t Nx);
	    ~poisson();


		void solve( double *data ) const noexcept;


	private:
			double Lx = 1;
			double Lx_inv = 1;
			size_t Nx = 0;

        	fftw_plan plan;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
        size_t alignment { 64 };

		poisson() = delete;
		poisson( const poisson  &rhs ) = delete;
		poisson(       poisson &&rhs ) = delete;
		poisson& operator=( const poisson  &rhs ) = delete;
		poisson& operator=(       poisson &&rhs ) = delete;
		poisson(float Lx, size_t Nx);
	    ~poisson();


		void solve( float *data ) const noexcept;

	private:
		float Lx = 1;
		float Lx_inv = 1;
		size_t Nx = 0;

		fftwf_plan plan;
	};

}

namespace dim2
{
	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
        const size_t alignment { 64 };

		poisson() = delete;
		poisson( const poisson  &rhs ) = delete;
		poisson(       poisson &&rhs ) = delete;
		poisson& operator=( const poisson  &rhs ) = delete;
		poisson& operator=(       poisson &&rhs ) = delete;
		poisson(double Lx, double Ly, size_t Nx, size_t Ny);
	    ~poisson();

		void solve( double *data ) const noexcept;


	private:
		double Lx = 1;
		double Lx_inv = 1;
		double Ly = 1;
		double Ly_inv = 1;
		size_t Nx = 0;
		size_t Ny = 0;

	    fftw_plan plan;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
	        size_t alignment { 64 };

		poisson() = delete;
		poisson( const poisson  &rhs ) = delete;
		poisson(       poisson &&rhs ) = delete;
		poisson& operator=( const poisson  &rhs ) = delete;
		poisson& operator=(       poisson &&rhs ) = delete;
		poisson(float Lx, float Ly, size_t Nx, size_t Ny);
	    ~poisson();

		void solve( float *data ) const noexcept;

	private:
		float Lx = 1;
		float Lx_inv = 1;
		float Ly = 1;
		float Ly_inv = 1;
		size_t Nx = 0;
		size_t Ny = 0;

		fftwf_plan plan;
	};

}

namespace dim3
{
	template <typename real> class poisson;

	// Defintion for real = double.
	template <>
	class poisson<double>
	{
	public:
        const size_t alignment { 64 };

		poisson() = delete;
        poisson( const poisson  &rhs ) = delete;
        poisson(       poisson &&rhs ) = delete;
        poisson& operator=( const poisson  &rhs ) = delete;
        poisson& operator=(       poisson &&rhs ) = delete;
		poisson(double Lx, double Ly, double Lz, size_t Nx, size_t Ny, size_t Nz);
	    ~poisson();

		void solve( double *data ) const noexcept;


	private:
		double Lx = 1;
		double Lx_inv = 1;
		double Ly = 1;
		double Ly_inv = 1;
		double Lz = 1;
		double Lz_inv = 1;
		size_t Nx = 0;
		size_t Ny = 0;
		size_t Nz = 0;

        fftw_plan plan;
	};

	// Declaration for real = float.
	template <>
	class poisson<float>
	{
	public:
        size_t alignment { 64 };

		poisson() = delete;
        poisson( const poisson  &rhs ) = delete;
        poisson(       poisson &&rhs ) = delete;
        poisson& operator=( const poisson  &rhs ) = delete;
        poisson& operator=(       poisson &&rhs ) = delete;
        poisson(float Lx, float Ly, float Lz, size_t Nx, size_t Ny, size_t Nz);
   	    ~poisson();

  		void solve( float *data ) const noexcept;


      	private:
        	float Lx = 1;
        	float Lx_inv = 1;
        	float Ly = 1;
        	float Ly_inv = 1;
        	float Lz = 1;
        	float Lz_inv = 1;
        	size_t Nx = 0;
        	size_t Ny = 0;
        	size_t Nz = 0;

        	fftwf_plan plan;
	};

}

}

#endif

