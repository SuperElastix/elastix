/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkRecursiveBSplineInterpolateImageFunctionWrapper_h
#define __itkRecursiveBSplineInterpolateImageFunctionWrapper_h

#include "itkAdvancedInterpolateImageFunction.h"

#include <vector>
#include "itkInterpolateImageFunction.h"
#include "vnl/vnl_matrix.h"

#include "itkBSplineDecompositionImageFilter.h"
#include "itkConceptChecking.h"
#include "itkCovariantVector.h"

namespace itk
{
/** \class RecursiveBSplineInterpolateImageFunctionWrapper
 * \brief Wrapper for the recursive B-Spline interpolation. This class is not templated over the spline order.
 *
 * This class serves as a wrapper for the RecursiveBSplineInterpolateImageFunction. That function is templated
 * over the spline order, but for backwards compatibility this wrapper is made, which is not templated over the
 * spline order, as the BsplineInterpolateImageFunction class.
 *
 * This class inherits from the class AdvancedInterpolateImageFunction, which is extended with virtual functions
 * of the BsplineInterpolateImageFunction class. In this class an instance is created of
 * the RecursiveBSplineInterpolateImageFunction class and all methods are accessed with this instance. Also, the
 * spline order is set in this class.
 *
 */

template< class TImageType,
          class TCoordRep = double,
          class TCoefficientType = double>
class ITK_EXPORT RecursiveBSplineInterpolateImageFunctionWrapper :
 public AdvancedInterpolateImageFunction< TImageType, TCoordRep >
{
public:
  /** Standard class typedefs. */
  typedef RecursiveBSplineInterpolateImageFunctionWrapper             Self;
  typedef AdvancedInterpolateImageFunction< TImageType, TCoordRep >   Superclass;
  typedef SmartPointer< Self >                                        Pointer;
  typedef SmartPointer< const Self >                                  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( RecursiveBSplineInterpolateImageFunctionWrapper, AdvancedInterpolateImageFunction );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  // MS: check that typedefs are really copied from superclass and not created new

  /** OutputType typedef support. */
  typedef typename Superclass::OutputType OutputType;

  /** InputImageType typedef support. */
  typedef typename Superclass::InputImageType InputImageType;

  /** Dimension underlying input image. */
  itkStaticConstMacro( ImageDimension, unsigned int, Superclass::ImageDimension );

  /** Index typedef support. */
  typedef typename Superclass::IndexType IndexType;

  /** ContinuousIndex typedef support. */
  typedef typename Superclass::ContinuousIndexType ContinuousIndexType;

  /** PointType typedef support */
  typedef typename Superclass::PointType PointType;

  /** Iterator typedef support */
  typedef ImageLinearIteratorWithIndex< TImageType > Iterator;

  /** Internal Coefficient typedef support */
  typedef TCoefficientType CoefficientDataType;
  typedef Image< CoefficientDataType,  itkGetStaticConstMacro(ImageDimension) >    CoefficientImageType;

  /** Define filter for calculating the BSpline coefficients */
  typedef BSplineDecompositionImageFilter< TImageType,    CoefficientImageType > CoefficientFilter;

  /** Derivative typedef support */
  typedef typename CoefficientFilter::Pointer    CoefficientFilterPointer;
  typedef CovariantVector< OutputType, itkGetStaticConstMacro(ImageDimension) >    CovariantVectorType;

  /** Evaluate the function at a ContinuousIndex position.
   *
   * Returns the B-Spline interpolated image intensity at a
   * specified point position. No bounds checking is done.
   * The point is assume to lie within the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method.
   */
  // MS: remove the ones with threadID??
  OutputType Evaluate(const PointType & point) const;
  OutputType Evaluate(const PointType & point, ThreadIdType threadID) const;
  OutputType EvaluateAtContinuousIndex(const ContinuousIndexType & index) const;
  OutputType EvaluateAtContinuousIndex(const ContinuousIndexType & index, ThreadIdType threadID) const;

  CovariantVectorType EvaluateDerivative(const PointType & point) const;
  CovariantVectorType EvaluateDerivative(const PointType & point, ThreadIdType threadID) const;
  CovariantVectorType EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x) const;
  CovariantVectorType EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x,ThreadIdType threadID) const;

  void EvaluateValueAndDerivative(const PointType & point, OutputType & value,CovariantVectorType & deriv) const;
  void EvaluateValueAndDerivative(const PointType & point, OutputType & value,CovariantVectorType & deriv,ThreadIdType threadID) const;
  void EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x, OutputType & value,CovariantVectorType & deriv) const;
  void EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x, OutputType & value,CovariantVectorType & deriv, ThreadIdType threadID) const;

  void SetSplineOrder(unsigned int splineOrder);
  void SetInputImage(const InputImageType *inputData);
  void SetNumberOfThreads(ThreadIdType numThreads);

  itkGetConstMacro(SplineOrder, int);

protected:
  RecursiveBSplineInterpolateImageFunctionWrapper();
  ~RecursiveBSplineInterpolateImageFunctionWrapper(){}

  typename AdvancedInterpolateImageFunction<TImageType, TCoordRep>::Pointer m_InterpolatorInstance;
  unsigned int m_SplineOrder;

  void PrintSelf(std::ostream & os, Indent indent) const;

  // MS: delete below, you never use them
  OutputType EvaluateAtContinuousIndexInternal(const ContinuousIndexType & index, vnl_matrix< long > & evaluateIndex, vnl_matrix< double > & weights) const
  {
    itkExceptionMacro ("Exception: the method 'EvaluateAtContinuousIndexInternal' is removed from this class.");
  }

  void EvaluateValueAndDerivativeAtContinuousIndexInternal(const ContinuousIndexType & x,OutputType & value, CovariantVectorType & derivativeValue, vnl_matrix< long > & evaluateIndex,vnl_matrix< double > & weights,  vnl_matrix< double > & weightsDerivative) const
  {
    itkExceptionMacro ("Exception: the method 'EvaluateValueAndDerivativeAtContinuousIndexInternal' is removed from this class.");
  }

  CovariantVectorType EvaluateDerivativeAtContinuousIndexInternal(const ContinuousIndexType & x, vnl_matrix< long > & evaluateIndex, vnl_matrix< double > & weights, vnl_matrix< double > & weightsDerivative) const
  {
    itkExceptionMacro ("Exception: the method 'EvaluateDerivativeAtContinuousIndexInternal' is removed from this class.");
  }


private:
  RecursiveBSplineInterpolateImageFunctionWrapper(const Self &); // purposely not implemented
  void operator=(const Self &);                                  // purposely not implemented
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRecursiveBSplineInterpolateImageFunctionWrapper.hxx"
#endif

#endif
