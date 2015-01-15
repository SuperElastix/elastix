/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkRecursiveBSplineInterpolateImageFunctionForITK_h
#define __itkRecursiveBSplineInterpolateImageFunctionForITK_h

#include "itkRecursiveBSplineInterpolateImageFunction.h"

namespace itk
{

/** \class RecursiveBSplineInterpolateImageFunctionForITK
 * \brief Evaluates the B-Spline interpolation of an image recursively.  Spline order may be from 0 to 5.
 *
 * Just adds threaded functions.
 *
 * This class defines N-Dimension B-Spline transformation, but in a recursive manner, such that it is much
 * faster than the itk class BSplineInterpolateImageFunction. This class is also templated over the spline order.
 * Inline classes are defined to obtain the B-spline weights for the chosen spline order. For backward compatibility
 * reasons a wrapper class was made, which is not templated over the spline order, called
 * RecursiveBSplineInterpolateImageFunctionWrapper.
 *
 * Computationally there are only round-off differences of order 1e-16 with the old ITK class, due to the fact that
 * floating point additions and multiplications are not commutative. Results are published in:
 *      W. Huizinga, S. Klein and D.H.J. Poot,
 *      "Fast Multidimensional B-spline Interpolation using Template Metaprogramming",
 *      Workshop on Biomedical Image Registration WBIR'14, July 2014, London.
 *
 * The B spline coefficients are calculated through the
 * BSplineDecompositionImageFilter
 *
 * Limitations:  Spline order must be between 0 and 5.
 *               Spline order must be set before setting the image.
 *               Uses mirror boundary conditions.
 *               Requires the same order of Spline for each dimension.
 *               Spline is determined in all dimensions, cannot selectively               <-- MS: do not understand this comment
 *                  pick dimension for calculating spline.
 */

template< class TImageType,
          class TCoordRep = double,
          class TCoefficientType = double,
          unsigned int SplineOrder = 3 >
class RecursiveBSplineInterpolateImageFunctionForITK :
  public RecursiveBSplineInterpolateImageFunction< TImageType, TCoordRep, TCoefficientType, SplineOrder >
{
public:
  /** Standard class typedefs. */
  typedef RecursiveBSplineInterpolateImageFunctionForITK                  Self;
  typedef AdvancedInterpolateImageFunction< TImageType, TCoordRep > Superclass;
  typedef SmartPointer< Self >                                      Pointer;
  typedef SmartPointer< const Self >                                ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( RecursiveBSplineInterpolateImageFunctionForITK, RecursiveBSplineInterpolateImageFunctionForITK );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** Dimension underlying input image. */
  itkStaticConstMacro( ImageDimension, unsigned int, Superclass::ImageDimension );

  /** typedef support. */
  typedef typename Superclass::OutputType           OutputType;
  typedef typename Superclass::InputImageType       InputImageType;
  typedef typename Superclass::IndexType            IndexType;
  typedef typename Superclass::ContinuousIndexType  ContinuousIndexType;
  typedef typename Superclass::PointType            PointType;
  typedef typename Superclass::CovariantVectorType  CovariantVectorType;

  // Threaded versions of the evaluate functions
  OutputType Evaluate( const PointType & point, ThreadIdType threadID ) const;
  OutputType EvaluateAtContinuousIndex(
    const ContinuousIndexType & cindex, ThreadIdType threadID ) const;
  CovariantVectorType EvaluateDerivative(
    const PointType & point, ThreadIdType threadID ) const;
  CovariantVectorType EvaluateDerivativeAtContinuousIndex(
    const ContinuousIndexType & cindex, ThreadIdType threadID ) const;
  void EvaluateValueAndDerivative(
    const PointType & point, OutputType & value, CovariantVectorType & deriv, ThreadIdType threadID ) const;
  void EvaluateValueAndDerivativeAtContinuousIndex(
    const ContinuousIndexType & x, OutputType & value, CovariantVectorType & deriv, ThreadIdType threadID ) const;

  // MS: delete?
  //void SetNumberOfThreads(ThreadIdType numThreads); // MS: delete??
  //itkGetConstMacro(NumberOfThreads, ThreadIdType); // MS: delete??

protected:
  RecursiveBSplineInterpolateImageFunctionForITK();
  ~RecursiveBSplineInterpolateImageFunctionForITK(){};

private:
  RecursiveBSplineInterpolateImageFunctionForITK(const Self &); // purposely not implemented
  void operator=(const Self &);                                 // purposely not implemented

};


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRecursiveBSplineInterpolateImageFunctionForITK.hxx"
#endif

#endif //__itkRecursiveBSplineInterpolateImageFunctionForITK_h
