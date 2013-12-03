/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBSplineInterpolationSecondOrderDerivativeWeightFunction_h
#define __itkBSplineInterpolationSecondOrderDerivativeWeightFunction_h

#include "itkBSplineInterpolationWeightFunctionBase.h"
#include "vnl/vnl_vector_fixed.h"

namespace itk
{

/** \class BSplineInterpolationSecondOrderDerivativeWeightFunction
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

template < class TCoordRep = float,
  unsigned int VSpaceDimension = 2,
  unsigned int VSplineOrder = 3 >
class BSplineInterpolationSecondOrderDerivativeWeightFunction :
  public BSplineInterpolationWeightFunctionBase<
  TCoordRep, VSpaceDimension, VSplineOrder >
{
public:
  /** Standard class typedefs. */
  typedef BSplineInterpolationSecondOrderDerivativeWeightFunction Self;
  typedef BSplineInterpolationWeightFunctionBase<
    TCoordRep, VSpaceDimension, VSplineOrder >      Superclass;
  typedef SmartPointer<Self>                        Pointer;
  typedef SmartPointer<const Self>                  ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( BSplineInterpolationSecondOrderDerivativeWeightFunction,
    BSplineInterpolationWeightFunctionBase );

  /** Space dimension. */
  itkStaticConstMacro( SpaceDimension, unsigned int, VSpaceDimension );

  /** Spline order. */
  itkStaticConstMacro( SplineOrder, unsigned int, VSplineOrder );

  /** Typedefs from Superclass. */
  typedef typename Superclass::WeightsType          WeightsType;
  typedef typename Superclass::IndexType            IndexType;
  typedef typename Superclass::SizeType             SizeType;
  typedef typename Superclass::ContinuousIndexType  ContinuousIndexType;

  /** Set the second order derivative directions. */
  virtual void SetDerivativeDirections( unsigned int dir0, unsigned int dir1 );

protected:
  BSplineInterpolationSecondOrderDerivativeWeightFunction();
  ~BSplineInterpolationSecondOrderDerivativeWeightFunction() {}

  /** Interpolation kernel types. */
  typedef typename Superclass::KernelType           KernelType;
  typedef typename Superclass::DerivativeKernelType DerivativeKernelType;
  typedef typename Superclass
    ::SecondOrderDerivativeKernelType               SecondOrderDerivativeKernelType;
  typedef typename Superclass::TableType            TableType;
  typedef typename Superclass::OneDWeightsType      OneDWeightsType;

  /** Compute the 1D weights, which are:
   * \f[ \beta( x[i] - startIndex[i] ), \beta( x[i] - startIndex[i] - 1 ),
   * \beta( x[i] - startIndex[i] - 2 ), \beta( x[i] - startIndex[i] - 3 ), \f]
   * with \f$\beta( x ) = \beta^2( x + 1/2 ) - \beta^2( x - 1/2 )\f$,
   * in case of non-equal derivative directions,
   * with \f$\beta( x ) = \beta^1( x + 1 ) - 2 * \beta^1( x ) + \beta^1( x - 1 ),\f$
   * in case of equal derivative directions,
   * with \f$\beta(x) = \beta^3(x)\f$ for the non-derivative directions.
   */
  virtual void Compute1DWeights(
    const ContinuousIndexType & index,
    const IndexType & startIndex,
    OneDWeightsType & weights1D ) const;

  /** Print the member variables. */
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

private:
  BSplineInterpolationSecondOrderDerivativeWeightFunction(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  vnl_vector_fixed< double, 2 >   m_DerivativeDirections;
  bool                            m_EqualDerivativeDirections;

};

} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBSplineInterpolationSecondOrderDerivativeWeightFunction.hxx"
#endif

#endif
