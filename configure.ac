AC_INIT([vlasovius],[0.1])
AC_CANONICAL_SYSTEM
AC_CONFIG_MACRO_DIR([m4])

AM_INIT_AUTOMAKE([-Wall -Werror nostdinc])
AM_SILENT_RULES([yes])
AM_PROG_AR

AC_LANG([C++])
AC_PROG_CC
AC_PROG_CXX
AC_OPENMP
AC_SUBST([AM_CXXFLAGS],["$OPENMP_CXXFLAGS"]) 

AX_BLAS([],[AC_MSG_ERROR([vlasovius requires BLAS.])])
AX_CXX_COMPILE_STDCXX_17([noext],[mandatory])
AX_CHECK_X86_FEATURES

AC_SEARCH_LIBS([__gmpz_init], [gmp], [],
  [AC_MSG_ERROR([GNU MP not found, see https://gmplib.org/])])


AC_SUBST([AM_CPPFLAGS],['-I$(top_srcdir)'])

AC_CONFIG_HEADERS([vlasovius/config.h])
AC_CONFIG_FILES([Makefile
                 vlasovius/makefile
                 vlasovius/kernels/makefile
                 vlasovius/geometry/makefile
                 vlasovius/poisson/makefile
                 bin/makefile])

LT_INIT
AC_OUTPUT

