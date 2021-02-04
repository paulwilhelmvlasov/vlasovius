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
AX_BOOST_BASE([1.61.0],,[AC_MSG_ERROR([vlasovius requires Boost.])])
AX_BOOST_REGEX


AX_CXX_COMPILE_STDCXX_11([noext],[mandatory])

AC_SUBST([AM_CPPFLAGS],['-I$(top_srcdir)'])

AC_CONFIG_HEADERS([vlasovius/config.h])
AC_CONFIG_FILES([Makefile
                 vlasovius/makefile
                 bin/makefile])

LT_INIT
AC_OUTPUT

