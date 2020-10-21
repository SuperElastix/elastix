// This is core/vnl/vnl_adjugate_fixed.h
#ifndef vnl_adjugate_fixed_h_
#define vnl_adjugate_fixed_h_
//:
// \file
// \brief Calculates adjugate or cofactor matrix of a small vnl_matrix_fixed.
// Code is only a small adaptation from Peter Vanroose's vnl_inverse.h
// adjugate == inverse/det
// cofactor == adjugate^T
// \author Floris Berendsen
// \date   18 April 2013
//
// \verbatim
// Code is only a small adaptation from Peter Vanroose's vnl_inverse.h
// \endverbatim

#include <vnl/vnl_matrix_fixed.h>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_det.h>

//: Calculates adjugate of a small vnl_matrix_fixed (not using svd)
//  This allows you to write e.g.
//
//  x = vnl_adjugate(A) * b;
//
// Note that this function is inlined (except for the call to vnl_det()),
// which makes it much faster than the vnl_matrix_adjugate_fixed class in vnl/algo
// since that one is using svd.
//
//  \relatesalso vnl_matrix_fixed

template <class T>
vnl_matrix_fixed<T, 1, 1> vnl_adjugate(vnl_matrix_fixed<T, 1, 1> const & m)
{
  return vnl_matrix_fixed<T, 1, 1>(m(0, 0));
}

//: Calculates adjugate of a small vnl_matrix_fixed (not using svd)
//  This allows you to write e.g.
//
//  x = vnl_adjugate(A) * b;
//
// Note that this function is inlined (except for the call to vnl_det()),
// which makes it much faster than the vnl_matrix_adjugate_fixed class in vnl/algo
// since that one is using svd.
//
//  \relatesalso vnl_matrix_fixed

template <class T>
vnl_matrix_fixed<T, 2, 2> vnl_adjugate(vnl_matrix_fixed<T, 2, 2> const & m)
{
  T d[4];
  d[0] = m(1, 1);
  d[1] = -m(0, 1);
  d[3] = m(0, 0);
  d[2] = -m(1, 0);
  return vnl_matrix_fixed<T, 2, 2>(d);
}

//: Calculates adjugate of a small vnl_matrix_fixed (not using svd)
//  This allows you to write e.g.
//
//  x = vnl_adjugate_fixed(A) * b;
//
// Note that this function is inlined (except for the call to vnl_det()),
// which makes it much faster than the vnl_matrix_adjugate_fixed class in vnl/algo
// since that one is using svd.
//
//  \relatesalso vnl_matrix_fixed

template <class T>
vnl_matrix_fixed<T, 3, 3> vnl_adjugate(vnl_matrix_fixed<T, 3, 3> const & m)
{
  T d[9];
  d[0] = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1));
  d[1] = (m(2, 1) * m(0, 2) - m(2, 2) * m(0, 1));
  d[2] = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1));
  d[3] = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2));
  d[4] = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0));
  d[5] = (m(1, 0) * m(0, 2) - m(1, 2) * m(0, 0));
  d[6] = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
  d[7] = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1));
  d[8] = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0));
  return vnl_matrix_fixed<T, 3, 3>(d);
}

//: Calculates adjugate_fixed of a small vnl_matrix_fixed (not using svd)
//  This allows you to write e.g.
//
//  x = vnl_adjugate_fixed(A) * b;
//
// Note that this function is inlined (except for the call to vnl_det()),
// which makes it much faster than the vnl_matrix_adjugate_fixed class in vnl/algo
// since that one is using svd.
//
//  \relatesalso vnl_matrix_fixed

template <class T>
vnl_matrix_fixed<T, 4, 4> vnl_adjugate(vnl_matrix_fixed<T, 4, 4> const & m)
{
  T d[16];
  d[0] = m(1, 1) * m(2, 2) * m(3, 3) - m(1, 1) * m(2, 3) * m(3, 2) - m(2, 1) * m(1, 2) * m(3, 3) +
         m(2, 1) * m(1, 3) * m(3, 2) + m(3, 1) * m(1, 2) * m(2, 3) - m(3, 1) * m(1, 3) * m(2, 2);
  d[1] = -m(0, 1) * m(2, 2) * m(3, 3) + m(0, 1) * m(2, 3) * m(3, 2) + m(2, 1) * m(0, 2) * m(3, 3) -
         m(2, 1) * m(0, 3) * m(3, 2) - m(3, 1) * m(0, 2) * m(2, 3) + m(3, 1) * m(0, 3) * m(2, 2);
  d[2] = m(0, 1) * m(1, 2) * m(3, 3) - m(0, 1) * m(1, 3) * m(3, 2) - m(1, 1) * m(0, 2) * m(3, 3) +
         m(1, 1) * m(0, 3) * m(3, 2) + m(3, 1) * m(0, 2) * m(1, 3) - m(3, 1) * m(0, 3) * m(1, 2);
  d[3] = -m(0, 1) * m(1, 2) * m(2, 3) + m(0, 1) * m(1, 3) * m(2, 2) + m(1, 1) * m(0, 2) * m(2, 3) -
         m(1, 1) * m(0, 3) * m(2, 2) - m(2, 1) * m(0, 2) * m(1, 3) + m(2, 1) * m(0, 3) * m(1, 2);
  d[4] = -m(1, 0) * m(2, 2) * m(3, 3) + m(1, 0) * m(2, 3) * m(3, 2) + m(2, 0) * m(1, 2) * m(3, 3) -
         m(2, 0) * m(1, 3) * m(3, 2) - m(3, 0) * m(1, 2) * m(2, 3) + m(3, 0) * m(1, 3) * m(2, 2);
  d[5] = m(0, 0) * m(2, 2) * m(3, 3) - m(0, 0) * m(2, 3) * m(3, 2) - m(2, 0) * m(0, 2) * m(3, 3) +
         m(2, 0) * m(0, 3) * m(3, 2) + m(3, 0) * m(0, 2) * m(2, 3) - m(3, 0) * m(0, 3) * m(2, 2);
  d[6] = -m(0, 0) * m(1, 2) * m(3, 3) + m(0, 0) * m(1, 3) * m(3, 2) + m(1, 0) * m(0, 2) * m(3, 3) -
         m(1, 0) * m(0, 3) * m(3, 2) - m(3, 0) * m(0, 2) * m(1, 3) + m(3, 0) * m(0, 3) * m(1, 2);
  d[7] = m(0, 0) * m(1, 2) * m(2, 3) - m(0, 0) * m(1, 3) * m(2, 2) - m(1, 0) * m(0, 2) * m(2, 3) +
         m(1, 0) * m(0, 3) * m(2, 2) + m(2, 0) * m(0, 2) * m(1, 3) - m(2, 0) * m(0, 3) * m(1, 2);
  d[8] = m(1, 0) * m(2, 1) * m(3, 3) - m(1, 0) * m(2, 3) * m(3, 1) - m(2, 0) * m(1, 1) * m(3, 3) +
         m(2, 0) * m(1, 3) * m(3, 1) + m(3, 0) * m(1, 1) * m(2, 3) - m(3, 0) * m(1, 3) * m(2, 1);
  d[9] = -m(0, 0) * m(2, 1) * m(3, 3) + m(0, 0) * m(2, 3) * m(3, 1) + m(2, 0) * m(0, 1) * m(3, 3) -
         m(2, 0) * m(0, 3) * m(3, 1) - m(3, 0) * m(0, 1) * m(2, 3) + m(3, 0) * m(0, 3) * m(2, 1);
  d[10] = m(0, 0) * m(1, 1) * m(3, 3) - m(0, 0) * m(1, 3) * m(3, 1) - m(1, 0) * m(0, 1) * m(3, 3) +
          m(1, 0) * m(0, 3) * m(3, 1) + m(3, 0) * m(0, 1) * m(1, 3) - m(3, 0) * m(0, 3) * m(1, 1);
  d[11] = -m(0, 0) * m(1, 1) * m(2, 3) + m(0, 0) * m(1, 3) * m(2, 1) + m(1, 0) * m(0, 1) * m(2, 3) -
          m(1, 0) * m(0, 3) * m(2, 1) - m(2, 0) * m(0, 1) * m(1, 3) + m(2, 0) * m(0, 3) * m(1, 1);
  d[12] = -m(1, 0) * m(2, 1) * m(3, 2) + m(1, 0) * m(2, 2) * m(3, 1) + m(2, 0) * m(1, 1) * m(3, 2) -
          m(2, 0) * m(1, 2) * m(3, 1) - m(3, 0) * m(1, 1) * m(2, 2) + m(3, 0) * m(1, 2) * m(2, 1);
  d[13] = m(0, 0) * m(2, 1) * m(3, 2) - m(0, 0) * m(2, 2) * m(3, 1) - m(2, 0) * m(0, 1) * m(3, 2) +
          m(2, 0) * m(0, 2) * m(3, 1) + m(3, 0) * m(0, 1) * m(2, 2) - m(3, 0) * m(0, 2) * m(2, 1);
  d[14] = -m(0, 0) * m(1, 1) * m(3, 2) + m(0, 0) * m(1, 2) * m(3, 1) + m(1, 0) * m(0, 1) * m(3, 2) -
          m(1, 0) * m(0, 2) * m(3, 1) - m(3, 0) * m(0, 1) * m(1, 2) + m(3, 0) * m(0, 2) * m(1, 1);
  d[15] = m(0, 0) * m(1, 1) * m(2, 2) - m(0, 0) * m(1, 2) * m(2, 1) - m(1, 0) * m(0, 1) * m(2, 2) +
          m(1, 0) * m(0, 2) * m(2, 1) + m(2, 0) * m(0, 1) * m(1, 2) - m(2, 0) * m(0, 2) * m(1, 1);
  return vnl_matrix_fixed<T, 4, 4>(d);
}

//: Calculates adjugate_fixed of a small vnl_matrix_fixed (not using svd)
//  This allows you to write e.g.
//
//  x = vnl_adjugate_fixed(A) * b;
//
// Note that this function is inlined (except for the call to vnl_det()),
// which makes it much faster than the vnl_matrix_adjugate_fixed class in vnl/algo
// since that one is using svd.
//
//  \relatesalso vnl_matrix

template <class T>
vnl_matrix<T>
vnl_adjugate_asfixed(vnl_matrix<T> const & m)
{
  assert(m.rows() == m.columns());
  assert(m.rows() <= 4);
  if (m.rows() == 1)
    return vnl_matrix<T>(1, 1, T(1) / m(0, 0));
  else if (m.rows() == 2)
    return vnl_adjugate(vnl_matrix_fixed<T, 2, 2>(m)).as_ref();
  else if (m.rows() == 3)
    return vnl_adjugate(vnl_matrix_fixed<T, 3, 3>(m)).as_ref();
  else
    return vnl_adjugate(vnl_matrix_fixed<T, 4, 4>(m)).as_ref();
}

//: Calculates transpose of the adjugate_fixed of a small vnl_matrix_fixed (not using svd)
//  This allows you to write e.g.
//
//  x = vnl_cofactor(A) * b;
//
// Note that this function is inlined (except for the call to vnl_det()),
// which makes it much faster than the vnl_matrix_adjugate_fixed class in vnl/algo
// since that one is using svd.  This is also faster than using
//
//  x = vnl_adjugate_fixed(A).transpose() * b;
//
//  \relatesalso vnl_matrix_fixed

template <class T>
vnl_matrix_fixed<T, 1, 1> vnl_cofactor(vnl_matrix_fixed<T, 1, 1> const & m)
{
  return vnl_matrix_fixed<T, 1, 1>(T(1) / m(0, 0));
}

//: Calculates transpose of the adjugate_fixed of a small vnl_matrix_fixed (not using svd)
//  This allows you to write e.g.
//
//  x = vnl_cofactor(A) * b;
//
// Note that this function is inlined (except for the call to vnl_det()),
// which makes it much faster than the vnl_matrix_adjugate_fixed class in vnl/algo
// since that one is using svd.  This is also faster than using
//
//  x = vnl_adjugate_fixed(A).transpose() * b;
//
//  \relatesalso vnl_matrix_fixed

template <class T>
vnl_matrix_fixed<T, 2, 2> vnl_cofactor(vnl_matrix_fixed<T, 2, 2> const & m)
{

  T d[4];
  d[0] = m(1, 1);
  d[2] = -m(0, 1);
  d[3] = m(0, 0);
  d[1] = -m(1, 0);
  return vnl_matrix_fixed<T, 2, 2>(d);
}

//: Calculates transpose of the adjugate_fixed of a small vnl_matrix_fixed (not using svd)
//  This allows you to write e.g.
//
//  x = vnl_cofactor(A) * b;
//
// Note that this function is inlined (except for the call to vnl_det()),
// which makes it much faster than the vnl_matrix_adjugate_fixed class in vnl/algo
// since that one is using svd.  This is also faster than using
//
//  x = vnl_adjugate_fixed(A).transpose() * b;
//
//  \relatesalso vnl_matrix_fixed

template <class T>
vnl_matrix_fixed<T, 3, 3> vnl_cofactor(vnl_matrix_fixed<T, 3, 3> const & m)
{

  T d[9];
  d[0] = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1));
  d[3] = (m(2, 1) * m(0, 2) - m(2, 2) * m(0, 1));
  d[6] = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1));
  d[1] = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2));
  d[4] = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0));
  d[7] = (m(1, 0) * m(0, 2) - m(1, 2) * m(0, 0));
  d[2] = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
  d[5] = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1));
  d[8] = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0));
  return vnl_matrix_fixed<T, 3, 3>(d);
}

//: Calculates transpose of the adjugate_fixed of a small vnl_matrix_fixed (not using svd)
//  This allows you to write e.g.
//
//  x = vnl_cofactor(A) * b;
//
// Note that this function is inlined (except for the call to vnl_det()),
// which makes it much faster than the vnl_matrix_adjugate_fixed class in vnl/algo
// since that one is using svd.  This is also faster than using
//
//  x = vnl_adjugate_fixed(A).transpose() * b;
//
//  \relatesalso vnl_matrix_fixed

template <class T>
vnl_matrix_fixed<T, 4, 4> vnl_cofactor(vnl_matrix_fixed<T, 4, 4> const & m)
{
  T d[16];
  d[0] = m(1, 1) * m(2, 2) * m(3, 3) - m(1, 1) * m(2, 3) * m(3, 2) - m(2, 1) * m(1, 2) * m(3, 3) +
         m(2, 1) * m(1, 3) * m(3, 2) + m(3, 1) * m(1, 2) * m(2, 3) - m(3, 1) * m(1, 3) * m(2, 2);
  d[4] = -m(0, 1) * m(2, 2) * m(3, 3) + m(0, 1) * m(2, 3) * m(3, 2) + m(2, 1) * m(0, 2) * m(3, 3) -
         m(2, 1) * m(0, 3) * m(3, 2) - m(3, 1) * m(0, 2) * m(2, 3) + m(3, 1) * m(0, 3) * m(2, 2);
  d[8] = m(0, 1) * m(1, 2) * m(3, 3) - m(0, 1) * m(1, 3) * m(3, 2) - m(1, 1) * m(0, 2) * m(3, 3) +
         m(1, 1) * m(0, 3) * m(3, 2) + m(3, 1) * m(0, 2) * m(1, 3) - m(3, 1) * m(0, 3) * m(1, 2);
  d[12] = -m(0, 1) * m(1, 2) * m(2, 3) + m(0, 1) * m(1, 3) * m(2, 2) + m(1, 1) * m(0, 2) * m(2, 3) -
          m(1, 1) * m(0, 3) * m(2, 2) - m(2, 1) * m(0, 2) * m(1, 3) + m(2, 1) * m(0, 3) * m(1, 2);
  d[1] = -m(1, 0) * m(2, 2) * m(3, 3) + m(1, 0) * m(2, 3) * m(3, 2) + m(2, 0) * m(1, 2) * m(3, 3) -
         m(2, 0) * m(1, 3) * m(3, 2) - m(3, 0) * m(1, 2) * m(2, 3) + m(3, 0) * m(1, 3) * m(2, 2);
  d[5] = m(0, 0) * m(2, 2) * m(3, 3) - m(0, 0) * m(2, 3) * m(3, 2) - m(2, 0) * m(0, 2) * m(3, 3) +
         m(2, 0) * m(0, 3) * m(3, 2) + m(3, 0) * m(0, 2) * m(2, 3) - m(3, 0) * m(0, 3) * m(2, 2);
  d[9] = -m(0, 0) * m(1, 2) * m(3, 3) + m(0, 0) * m(1, 3) * m(3, 2) + m(1, 0) * m(0, 2) * m(3, 3) -
         m(1, 0) * m(0, 3) * m(3, 2) - m(3, 0) * m(0, 2) * m(1, 3) + m(3, 0) * m(0, 3) * m(1, 2);
  d[13] = m(0, 0) * m(1, 2) * m(2, 3) - m(0, 0) * m(1, 3) * m(2, 2) - m(1, 0) * m(0, 2) * m(2, 3) +
          m(1, 0) * m(0, 3) * m(2, 2) + m(2, 0) * m(0, 2) * m(1, 3) - m(2, 0) * m(0, 3) * m(1, 2);
  d[2] = m(1, 0) * m(2, 1) * m(3, 3) - m(1, 0) * m(2, 3) * m(3, 1) - m(2, 0) * m(1, 1) * m(3, 3) +
         m(2, 0) * m(1, 3) * m(3, 1) + m(3, 0) * m(1, 1) * m(2, 3) - m(3, 0) * m(1, 3) * m(2, 1);
  d[6] = -m(0, 0) * m(2, 1) * m(3, 3) + m(0, 0) * m(2, 3) * m(3, 1) + m(2, 0) * m(0, 1) * m(3, 3) -
         m(2, 0) * m(0, 3) * m(3, 1) - m(3, 0) * m(0, 1) * m(2, 3) + m(3, 0) * m(0, 3) * m(2, 1);
  d[10] = m(0, 0) * m(1, 1) * m(3, 3) - m(0, 0) * m(1, 3) * m(3, 1) - m(1, 0) * m(0, 1) * m(3, 3) +
          m(1, 0) * m(0, 3) * m(3, 1) + m(3, 0) * m(0, 1) * m(1, 3) - m(3, 0) * m(0, 3) * m(1, 1);
  d[14] = -m(0, 0) * m(1, 1) * m(2, 3) + m(0, 0) * m(1, 3) * m(2, 1) + m(1, 0) * m(0, 1) * m(2, 3) -
          m(1, 0) * m(0, 3) * m(2, 1) - m(2, 0) * m(0, 1) * m(1, 3) + m(2, 0) * m(0, 3) * m(1, 1);
  d[3] = -m(1, 0) * m(2, 1) * m(3, 2) + m(1, 0) * m(2, 2) * m(3, 1) + m(2, 0) * m(1, 1) * m(3, 2) -
         m(2, 0) * m(1, 2) * m(3, 1) - m(3, 0) * m(1, 1) * m(2, 2) + m(3, 0) * m(1, 2) * m(2, 1);
  d[7] = m(0, 0) * m(2, 1) * m(3, 2) - m(0, 0) * m(2, 2) * m(3, 1) - m(2, 0) * m(0, 1) * m(3, 2) +
         m(2, 0) * m(0, 2) * m(3, 1) + m(3, 0) * m(0, 1) * m(2, 2) - m(3, 0) * m(0, 2) * m(2, 1);
  d[11] = -m(0, 0) * m(1, 1) * m(3, 2) + m(0, 0) * m(1, 2) * m(3, 1) + m(1, 0) * m(0, 1) * m(3, 2) -
          m(1, 0) * m(0, 2) * m(3, 1) - m(3, 0) * m(0, 1) * m(1, 2) + m(3, 0) * m(0, 2) * m(1, 1);
  d[15] = m(0, 0) * m(1, 1) * m(2, 2) - m(0, 0) * m(1, 2) * m(2, 1) - m(1, 0) * m(0, 1) * m(2, 2) +
          m(1, 0) * m(0, 2) * m(2, 1) + m(2, 0) * m(0, 1) * m(1, 2) - m(2, 0) * m(0, 2) * m(1, 1);
  return vnl_matrix_fixed<T, 4, 4>(d);
}

//: Calculates transpose of the adjugate_fixed of a small vnl_matrix_fixed (not using svd)
//  This allows you to write e.g.
//
//  x = vnl_cofactor(A) * b;
//
// Note that this function is inlined (except for the call to vnl_det()),
// which makes it much faster than the vnl_matrix_adjugate_fixed class in vnl/algo
// since that one is using svd.  This is also faster than using
//
//  x = vnl_adjugate_fixed(A).transpose() * b;
//
//  \relatesalso vnl_matrix

template <class T>
vnl_matrix<T>
vnl_cofactor(vnl_matrix<T> const & m)
{
  assert(m.rows() == m.columns());
  assert(m.rows() <= 4);
  if (m.rows() == 1)
    return vnl_matrix<T>(1, 1, T(1) / m(0, 0));
  else if (m.rows() == 2)
    return vnl_cofactor(vnl_matrix_fixed<T, 2, 2>(m)).as_ref();
  else if (m.rows() == 3)
    return vnl_cofactor(vnl_matrix_fixed<T, 3, 3>(m)).as_ref();
  else
    return vnl_cofactor(vnl_matrix_fixed<T, 4, 4>(m)).as_ref();
}


template <class T>
vnl_vector_fixed<T, 3> vnl_cofactor_row1(vnl_vector_fixed<T, 3> const & row2, vnl_vector_fixed<T, 3> const & row3)
{
  T d[3];
  d[0] = (row2[1] * row3[2] - row2[2] * row3[1]);
  d[1] = (row2[2] * row3[0] - row2[0] * row3[2]);
  d[2] = (row2[0] * row3[1] - row2[1] * row3[0]);
  return vnl_vector_fixed<T, 3>(d);
}

#endif // vnl_adjugate_fixed_h_
