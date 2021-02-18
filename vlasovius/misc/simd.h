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
#ifndef VLASOVIUS_MISC_SIMD_H
#define VLASOVIUS_MISC_SIMD_H

#include <vlasovius/config.h>



namespace vlasovius
{

namespace misc
{

enum class simd_abi { scalar, avx };

#if defined(__AVX2__)
constexpr simd_abi native_simd_abi { simd_abi::avx };
#else
constexpr simd_abi native_simd_abi { simd_abi::scalar };
#endif


template <typename T, simd_abi abi = native_simd_abi > class simd;

template <typename T>
using native_simd = simd<T,native_simd_abi>;

}

}

#if defined(__AVX2__)
#include <vlasovius/misc/simd_avx.tpp>
#endif

#include <vlasovius/misc/simd_scalar.tpp>

#endif

