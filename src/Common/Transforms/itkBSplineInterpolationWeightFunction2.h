/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBSplineInterpolationWeightFunction2_h
#define __itkBSplineInterpolationWeightFunction2_h

#include "itkBSplineInterpolationWeightFunctionBase.h"

namespace itk
{

/** \class BSplineInterpolationWeightFunction2
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
class BSplineInterpolationWeightFunction2 :
  public BSplineInterpolationWeightFunctionBase<
  TCoordRep, VSpaceDimension, VSplineOrder >
{
public:

  /** Standard class typedefs. */
  typedef BSplineInterpolationWeightFunction2 Self;
  typedef BSplineInterpolationWeightFunctionBase<
    TCoordRep, VSpaceDimension, VSplineOrder >      Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** New macro for creation of through the object factory.*/
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( BSplineInterpolationWeightFunction2,
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

protected:

  BSplineInterpolationWeightFunction2();
  ~BSplineInterpolationWeightFunction2() {}

  /** Interpolation kernel types. */
  typedef typename Superclass::KernelType           KernelType;
  typedef typename Superclass::DerivativeKernelType DerivativeKernelType;
  typedef typename Superclass
    ::SecondOrderDerivativeKernelType SecondOrderDerivativeKernelType;
  typedef typename Superclass::TableType       TableType;
  typedef typename Superclass::OneDWeightsType OneDWeightsType;
  typedef typename Superclass::WeightArrayType WeightArrayType;

  /* Compute the 1D weights, which are:
   * [ \beta^3( x[i] - startIndex[i] ), \beta^3( x[i] - startIndex[i] - 1 ),
   * \beta^3( x[i] - startIndex[i] - 2 ), \beta^3( x[i] - startIndex[i] - 3 ) ]
   */
  virtual void Compute1DWeights(
    const ContinuousIndexType & index,
    const IndexType & startIndex,
    OneDWeightsType & weights1D ) const;

private:

  BSplineInterpolationWeightFunction2( const Self & ); // purposely not implemented
  void operator=( const Self & );                      // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBSplineInterpolationWeightFunction2.hxx"
#endif

#endif
