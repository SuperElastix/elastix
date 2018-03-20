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
#ifndef __itkRecursiveBSplineInterpolationWeightFunction_h
#define __itkRecursiveBSplineInterpolationWeightFunction_h

#include "itkBSplineInterpolationWeightFunction.h"

#include "itkBSplineKernelFunction2.h"
#include "itkBSplineDerivativeKernelFunction2.h"
#include "itkBSplineSecondOrderDerivativeKernelFunction2.h"

namespace itk
{
/** Recursive template to retrieve the number of B-spline indices at compile time. */
template< unsigned int SplineOrder, unsigned int Dimension >
class GetConstNumberOfIndicesHack
{
public:

  typedef GetConstNumberOfIndicesHack< SplineOrder, Dimension - 1 > OneDimensionLess;
  itkStaticConstMacro( Value, unsigned int, ( SplineOrder + 1 ) * OneDimensionLess::Value );
};

template< unsigned int SplineOrder >
class GetConstNumberOfIndicesHack< SplineOrder, 0 >
{
public:

  itkStaticConstMacro( Value, unsigned int, 1 );
};

/** Recursive template to retrieve the number of B-spline weights at compile time. */
template< unsigned int SplineOrder, unsigned int Dimension >
class GetConstNumberOfWeightsHackRecursiveBSpline
{
public:

  itkStaticConstMacro( Value, unsigned int, ( SplineOrder + 1 ) * Dimension  );
};

/** \class RecursiveBSplineInterpolationWeightFunction
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
 * \ingroup ITKCommon
 */
template<
  typename TCoordRep           = float,
  unsigned int VSpaceDimension = 2,
  unsigned int VSplineOrder    = 3 >
class RecursiveBSplineInterpolationWeightFunction :
  public BSplineInterpolationWeightFunction< TCoordRep, VSpaceDimension, VSplineOrder >
{
public:

  /** Standard class typedefs. */
  typedef RecursiveBSplineInterpolationWeightFunction Self;
  typedef BSplineInterpolationWeightFunction<
    TCoordRep, VSpaceDimension, VSplineOrder >        Superclass;
  typedef SmartPointer< Self >                        Pointer;
  typedef SmartPointer< const Self >                  ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( RecursiveBSplineInterpolationWeightFunction, FunctionBase );

  /** Space dimension. */
  itkStaticConstMacro( SpaceDimension, unsigned int, VSpaceDimension );

  /** Spline order. */
  itkStaticConstMacro( SplineOrder, unsigned int, VSplineOrder );

  /** Typedefs from superclass*/
  typedef typename Superclass::WeightsType         WeightsType;
  typedef typename Superclass::IndexType           IndexType;
  typedef typename Superclass::SizeType            SizeType;
  typedef typename Superclass::ContinuousIndexType ContinuousIndexType;
  //typedef typename Superclass::

  /** Get number of hacks. */
  typedef GetConstNumberOfWeightsHackRecursiveBSpline<
    itkGetStaticConstMacro( SplineOrder ),
    itkGetStaticConstMacro( SpaceDimension ) > GetConstNumberOfWeightsHackRecursiveBSplineType;
  itkStaticConstMacro( NumberOfWeights, unsigned int, GetConstNumberOfWeightsHackRecursiveBSplineType::Value );
  typedef GetConstNumberOfIndicesHack<
    itkGetStaticConstMacro( SplineOrder ),
    itkGetStaticConstMacro( SpaceDimension ) > GetConstNumberOfIndicesHackType;
  itkStaticConstMacro( NumberOfIndices, unsigned int, GetConstNumberOfIndicesHackType::Value );

  /** Get number of weights. */
  itkGetConstMacro( NumberOfWeights, unsigned int );

  /** Get number of indices. */
  itkGetConstMacro( NumberOfIndices, unsigned int );

  /** Evaluate the weights at specified ContinousIndex position.
   * Subclasses must provide this method. */
  virtual WeightsType Evaluate( const ContinuousIndexType & index ) const;

  /** Evaluate the weights at specified ContinousIndex position.
   * The weights are returned in the user specified container.
   * This function assume that weights can hold
   * (SplineOrder + 1)^(SpaceDimension) elements. For efficiency,
   * no size checking is done.
   * On return, startIndex contains the start index of the
   * support region over which the weights are defined.
   */
  virtual void Evaluate( const ContinuousIndexType & index,
    WeightsType & weights, IndexType & startIndex ) const;

  void EvaluateDerivative( const ContinuousIndexType & index,
    WeightsType & weights, const IndexType & startIndex ) const;

  void EvaluateSecondOrderDerivative( const ContinuousIndexType & index,
    WeightsType & weights, const IndexType & startIndex ) const;

protected:

  RecursiveBSplineInterpolationWeightFunction();
  ~RecursiveBSplineInterpolationWeightFunction() {}
  void PrintSelf( std::ostream & os, Indent indent ) const;

private:

  RecursiveBSplineInterpolationWeightFunction( const Self & ); // purposely not implemented
  void operator=( const Self & );                              // purposely not implemented

  /** Private members; We unfortunatly cannot use those of the superclass. */
  unsigned int m_NumberOfWeights;
  unsigned int m_NumberOfIndices;
  SizeType     m_SupportSize;

  /** Interpolation kernel type. */
  typedef BSplineKernelFunction2< itkGetStaticConstMacro( SplineOrder ) >                      KernelType;
  typedef BSplineDerivativeKernelFunction2< itkGetStaticConstMacro( SplineOrder ) >            DerivativeKernelType;
  typedef BSplineSecondOrderDerivativeKernelFunction2< itkGetStaticConstMacro( SplineOrder ) > SecondOrderDerivativeKernelType;

  /** Interpolation kernel. */
  typename KernelType::Pointer m_Kernel;
  typename DerivativeKernelType::Pointer m_DerivativeKernel;
  typename SecondOrderDerivativeKernelType::Pointer m_SecondOrderDerivativeKernel;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRecursiveBSplineInterpolationWeightFunction.hxx"
#endif

#endif
