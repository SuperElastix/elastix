/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
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
