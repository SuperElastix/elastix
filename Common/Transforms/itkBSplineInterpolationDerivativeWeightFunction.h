/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBSplineInterpolationDerivativeWeightFunction_h
#define __itkBSplineInterpolationDerivativeWeightFunction_h

#include "itkBSplineInterpolationWeightFunctionBase.h"

namespace itk
{

/** \class BSplineInterpolationDerivativeWeightFunction
 * \brief Returns the weights over the support region used for B-spline
 * interpolation/reconstruction.
 *
 * Computes/evaluate the B-spline interpolation weights over the
 * support region of the B-spline.
 *
 * This class is templated over the coordinate representation type,
 * the space dimension and the spline order.
 *
 * \sa Point
 * \sa Index
 * \sa ContinuousIndex
 *
 * \ingroup Functions ImageInterpolators
 */

template< class TCoordRep    = float,
unsigned int VSpaceDimension = 2,
unsigned int VSplineOrder    = 3 >
class BSplineInterpolationDerivativeWeightFunction :
  public BSplineInterpolationWeightFunctionBase<
  TCoordRep, VSpaceDimension, VSplineOrder >
{
public:

  /** Standard class typedefs. */
  typedef BSplineInterpolationDerivativeWeightFunction Self;
  typedef BSplineInterpolationWeightFunctionBase<
    TCoordRep, VSpaceDimension, VSplineOrder >      Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( BSplineInterpolationDerivativeWeightFunction,
    BSplineInterpolationWeightFunctionBase );

  /** Space dimension. */
  itkStaticConstMacro( SpaceDimension, unsigned int, VSpaceDimension );

  /** Spline order. */
  itkStaticConstMacro( SplineOrder, unsigned int, VSplineOrder );

  /** Typedefs from Superclass. */
  typedef typename Superclass::WeightsType         WeightsType;
  typedef typename Superclass::IndexType           IndexType;
  typedef typename Superclass::SizeType            SizeType;
  typedef typename Superclass::ContinuousIndexType ContinuousIndexType;

  /** Set the first order derivative direction. */
  virtual void SetDerivativeDirection( unsigned int dir );

protected:

  BSplineInterpolationDerivativeWeightFunction();
  ~BSplineInterpolationDerivativeWeightFunction() {}

  /** Interpolation kernel types. */
  typedef typename Superclass::KernelType           KernelType;
  typedef typename Superclass::DerivativeKernelType DerivativeKernelType;
  typedef typename Superclass
    ::SecondOrderDerivativeKernelType SecondOrderDerivativeKernelType;
  typedef typename Superclass::TableType       TableType;
  typedef typename Superclass::OneDWeightsType OneDWeightsType;

  /** Compute the 1D weights, which are:
   * \f[ \beta( x[i] - startIndex[i] ), \beta( x[i] - startIndex[i] - 1 ),
   * \beta( x[i] - startIndex[i] - 2 ), \beta( x[i] - startIndex[i] - 3 ) \f],
   * with \f$\beta( x ) = \beta^2( x + 1/2 ) - \beta^2( x - 1/2 )\f$, in case of the
   * derivative direction, and just \f$\beta(x) = \beta^3(x)\f$ for the non-derivative
   * directions.
   */
  virtual void Compute1DWeights(
    const ContinuousIndexType & index,
    const IndexType & startIndex,
    OneDWeightsType & weights1D ) const;

  /** Print the member variables. */
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

private:

  BSplineInterpolationDerivativeWeightFunction( const Self & ); // purposely not implemented
  void operator=( const Self & );                               // purposely not implemented

  /** Member variables. */
  unsigned int m_DerivativeDirection;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBSplineInterpolationDerivativeWeightFunction.hxx"
#endif

#endif
