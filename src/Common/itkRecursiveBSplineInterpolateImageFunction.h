/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkRecursiveBSplineInterpolateImageFunction_h
#define __itkRecursiveBSplineInterpolateImageFunction_h

#include "itkAdvancedInterpolateImageFunction.h"

#include <vector>
#include "itkImageLinearIteratorWithIndex.h"
#include "vnl/vnl_matrix.h"
#include "itkBSplineDecompositionImageFilter.h"
#include "itkConceptChecking.h"
#include "itkCovariantVector.h"


namespace itk
{
/** Declaration of inline class BSplineWeights */
template< unsigned int SplineOrder, class TCoefficientType = double >
class BSplineWeights; // MS: move to separate file?


/** \class RecursiveBSplineInterpolateImageFunction
 * \brief Evaluates the B-Spline interpolation of an image recursively.  Spline order may be from 0 to 5.
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
class RecursiveBSplineInterpolateImageFunction :
  public AdvancedInterpolateImageFunction< TImageType, TCoordRep >
{
public:
  /** Standard class typedefs. */
  typedef RecursiveBSplineInterpolateImageFunction                  Self;
  typedef AdvancedInterpolateImageFunction< TImageType, TCoordRep > Superclass;
  typedef SmartPointer< Self >                                      Pointer;
  typedef SmartPointer< const Self >                                ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( RecursiveBSplineInterpolateImageFunction, AdvancedInterpolateImageFunction );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

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
  typedef Image< CoefficientDataType, itkGetStaticConstMacro( ImageDimension ) >   CoefficientImageType;

  /** Define filter for calculating the BSpline coefficients */
  typedef BSplineDecompositionImageFilter< TImageType, CoefficientImageType > CoefficientFilter;
  typedef typename CoefficientFilter::Pointer    CoefficientFilterPointer;

  /** Derivative typedef support */
  //typedef CovariantVector< OutputType, itkGetStaticConstMacro(ImageDimension) >  CovariantVectorType;
  typedef typename Superclass::CovariantVectorType CovariantVectorType;

  /** Evaluate the function at a ContinuousIndex position.
   *
   * Returns the B-Spline interpolated image intensity at a
   * specified point position. No bounds checking is done.
   * The point is assume to lie within the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method.
   */
  OutputType Evaluate( const PointType & point ) const;

  OutputType EvaluateAtContinuousIndex( const ContinuousIndexType & index ) const;

  CovariantVectorType EvaluateDerivative( const PointType & point) const;

  CovariantVectorType EvaluateDerivativeAtContinuousIndex( const ContinuousIndexType & x ) const;

  void EvaluateValueAndDerivative(
    const PointType & point, OutputType & value, CovariantVectorType & deriv ) const;

  void EvaluateValueAndDerivativeAtContinuousIndex(
    const ContinuousIndexType & x, OutputType & value, CovariantVectorType & deriv ) const;

  /** The UseImageDirection flag determines whether image derivatives are
   * computed with respect to the image grid or with respect to the physical
   * space. When this flag is ON the derivatives are computed with respect to
   * the coordinate system of physical space. The difference is whether we take
   * into account the image Direction or not. The flag ON will take into
   * account the image direction and will result in an extra matrix
   * multiplication compared to the amount of computation performed when the
   * flag is OFF.
   * The default value of this flag is On.
   */
  itkSetMacro( UseImageDirection, bool );
  itkGetConstMacro( UseImageDirection, bool );
  itkBooleanMacro( UseImageDirection );

  /** Set the input image, also in the coefficient filter. */
  void SetInputImage( const TImageType * inputData );

protected:
  RecursiveBSplineInterpolateImageFunction();
  ~RecursiveBSplineInterpolateImageFunction(){};

  void PrintSelf( std::ostream & os, Indent indent ) const;

  std::vector< CoefficientDataType >          m_Scratch;
  typename TImageType::SizeType               m_DataLength;
  typename CoefficientImageType::ConstPointer m_Coefficients;

private:
  RecursiveBSplineInterpolateImageFunction(const Self &); // purposely not implemented
  void operator=(const Self &);                           // purposely not implemented

  /** Determines the weights for interpolation of the value x */
  void SetInterpolationWeights( const ContinuousIndexType & x, const vnl_matrix< long > & evaluateIndex, double weights[] ) const;

  /** Determines the weights for the derivative portion of the value x */
  void SetDerivativeWeights( const ContinuousIndexType & x, const vnl_matrix< long > & evaluateIndex, double weights[] ) const;

  /** Determines the weights for the hessian portion of the value x */
  //void SetHessianWeights(const ContinuousIndexType & x, const vnl_matrix< long > & evaluateIndex, double weights[]) const;

  /** Determines the indices to use give the splines region of support. */
  void DetermineRegionOfSupport( vnl_matrix< long > & evaluateIndex,  const ContinuousIndexType & x ) const;

  /** Set the indices in evaluateIndex at the boundaries based on mirror
   * boundary conditions.
   */
  void ApplyMirrorBoundaryConditions( vnl_matrix< long > & evaluateIndex ) const;

  /** Maximum number of iterations points == (splineOrder+1)*ImageDimension */
  unsigned long m_MaxNumberInterpolationPoints; // MS: do you use this variable??
  // Or do you mean:
  //itkStaticConstMacro( MaxNumberInterpolationPoints, unsigned int, (ImageDimension + 1) * SplineOrder );

  CoefficientFilterPointer m_CoefficientFilter;

  /** flag to take or not the image direction into account when computing the
   * derivatives.
   */
  bool m_UseImageDirection;

  /** Member variable containing the table with step sizes to take in each dimension
   * to obtain the correct index of the coefficient (vector alpha in the mentioned paper).
   */
  IndexType m_OffsetTable;

  typename InputImageType::SpacingType m_Spacing;
  InputImageType * m_inputImage;
};

/** Recursive sampling functions, templated over image dimension.
 * The sample function takes in the coefficient, the step vector kappa and the weights vector,
 * containing the B-spline weights per dimension and per spline order index.
 * The function is called in the method EvaluateAtContinuousIndex(x).
 * The output is the interpolated value at x.
 * The sampleValueAndDerivative function outputs both the interpolated value at x as well as
 * the derivative evaluated at x.
 */
template< unsigned int Dimension, unsigned int SplineOrder, class TCoordRep >
class SampleFunction
{
public :

  /** briefly explain */
  static inline TCoordRep SampleValue( const TCoordRep * source,
    const long * steps,
    const double * weights )
  {
    // MS: a bit more comments explaining the code, perhaps with references to the paper
    const unsigned int helper = ( Dimension - 1 ) * ( SplineOrder + 1 );
    TCoordRep value = 0.0;
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      const TCoordRep * a = source + steps[ k + helper ];
      value += SampleFunction< Dimension - 1, SplineOrder, TCoordRep>::
        SampleValue( a, steps, weights ) * weights[ k + helper ];
    }
    return value;
  } // end SampleValue()


  /** briefly explain */
  static inline void SampleValueAndDerivative( TCoordRep derivativeAndValue[],
    const TCoordRep * source,
    const long * steps,
    const double * weights,
    const double * derivativeWeights )
  {
    /** derivativeAndValue length must be at least dim + 1. */
    TCoordRep derivativeAndValueNext[ Dimension + 1 ];
    const unsigned int helper = ( Dimension - 1 ) * ( SplineOrder + 1 );

    // MS: a bit more comments explaining the code, perhaps with references to the paper
    for( unsigned int n = 0; n <= Dimension; ++n )
    {
      derivativeAndValue[ n ] = 0.0;
      //derivativeAndValueNext[ n ] = 0.0; // TODO: check that this can be removed
    }

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      const TCoordRep * a = source + steps[ k + helper ];

      SampleFunction< Dimension - 1, SplineOrder, TCoordRep >::
        SampleValueAndDerivative( derivativeAndValueNext, a, steps, weights, derivativeWeights );
      for( unsigned int n = 0; n < Dimension; ++n )
      {
        derivativeAndValue[ n ] += derivativeAndValueNext[ n ] * weights[ k + helper ];
      }
      derivativeAndValue[ Dimension ] += derivativeAndValueNext[ 0 ] * derivativeWeights[ k + helper ];
    }

  } // end SampleValueAndDerivative()

  //    static inline void sampleHessian( TCoordRep derivativeAndValue[],
  //                                      const TCoordRep * source,
  //                                      const long * steps,
  //                                      const double * weights,
  //                                      const double * derivativeWeights,
  //                                      const double * hessianWeights)
  //    {
  //        /** hessianAndDerivativeAndValue length must be at least 3*dim+1
  //          */

  //        for(unsigned int n= 0; n <= 2*dim; ++n)
  //        {
  //            hessian[n] = 0.0;
  //        }

  //        TCoordRep hessianNext[2*dim];

  //        for (unsigned int k = 0; k <= splineOrder; k++)
  //        {
  //            const TCoordRep * a = source + steps[ k + (dim-1)*(splineOrder+1) ];

  //            sampleFunction<dim-1, splineOrder, TCoordRep>::
  //                    sampleHessian(hessianNext, a, steps, weights, derivativeWeights, hessianWeights);
  //            for(unsigned int n = 0; n < dim; ++n)
  //            {
  //                hessian[n] += hessianNext[n]*weights[ k + (dim-1)*(splineOrder+1) ];

  //            }
  //            hessian[dim] += hessianNext[0]*
  //                    hessianWeights[ k + (dim-1)*(splineOrder+1) ];
  //        }


  //    }
}; // end class SampleFunction

/** End cases of the sample functions. A pointer to the coefficients is returned. */
template< unsigned int SplineOrder, class TCoordRep >
class SampleFunction< 0, SplineOrder, TCoordRep >
{
public:
  static inline TCoordRep SampleValue(
    const TCoordRep * source, const long * steps, const double * weights )
  {
    return (*source);
  } // end SampleValue()

  static inline void SampleValueAndDerivative(
    TCoordRep derivativeAndValue[],
    const TCoordRep * source,
    const long * steps,
    const double * weights,
    const double * derivativeWeights)
  {
    derivativeAndValue[ 0 ] = *source;
  } // end SampleValueAndDerivative()
};


// MS: move below to separate files?

/** Specializations for the BSplineWeights classes, for each spline order.
 * All classes contain a GetWeights method and a GetDerivativeWeights method.
 * These methods are called by SetInterpolationWeights and
 * and SetDerivativeWeights.
 */

// Specialization for B-spline order 0: Nearest Neighbor interpolation
template< class TCoefficientType >
class BSplineWeights< 0, TCoefficientType >
{
public:
  static inline void GetWeights( itk::Vector<double,1> & bsplineWeights,
    const TCoefficientType & w )
  {
    bsplineWeights[ 0 ] = 1.0;
  } // end GetWeights()

  static inline void GetDerivativeWeights( itk::Vector<double,1> & bsplineWeightsD,
    const TCoefficientType & w )
  {
    bsplineWeightsD[ 0 ] = 0.0;
    // MS: throw exception instead of print error
    std::cerr << "Error: Cannot compute derivative of 0th order B-Spline"
      << std::endl;
    // itkExceptionMacro( << "ERROR: Cannot compute derivative of 0th order B-spline" );
  } // end GetDerivativeWeights()

  //    static inline void
  //    getHessianWeights( itk::Vector<double,1> &bsplweightsH, const TCoefficientType & w)
  //    {
  //        bsplweightsH[0] = 0.0;
  //        std::cerr << "Error: Cannot compute hessian of 0th order B-Spline"
  //                  << std::endl;

  //    }
};

// Specialization for B-spline order 1: Linear interpolation
template< class TCoefficientType >
class BSplineWeights< 1, TCoefficientType >
{
public:
  static inline void GetWeights( itk::Vector<double,2> & bsplineWeights,
    const TCoefficientType & w )
  {
    bsplineWeights[ 0 ] = 1.0 - w;
    bsplineWeights[ 1 ] = w;
  } // end GetWeights()

  static inline void GetDerivativeWeights( itk::Vector<double,2> & bsplineWeightsD,
    const TCoefficientType & w )
  {
    bsplineWeightsD[ 0 ] = -1.0;
    bsplineWeightsD[ 1 ] = 1.0;
  } // end GetDerivativeWeights()

  //    static inline void
  //    getHessianWeights( itk::Vector<double,2> &bsplweightsH,
  //                       const TCoefficientType & w)
  //    {
  //        bsplweightsH[0] = 0.0;
  //        bsplweightsH[1] = 0.0;
  //        std::cerr << "Error: Cannot compute hessian of 1st order B-Spline"
  //                  << std::endl;

  //    }
};

// Specialization for B-spline order 2: Quadratic interpolation
template< class TCoefficientType >
class BSplineWeights< 2, TCoefficientType >
{
public:
  static inline void GetWeights( itk::Vector<double,3> & bsplineWeights,
    const TCoefficientType & w )
  {
    bsplineWeights[ 1 ] = 0.75 - w * w;
    bsplineWeights[ 2 ] = 0.5 * ( w - bsplineWeights[ 1 ] + 1.0 );
    bsplineWeights[ 0 ] = 1.0 - bsplineWeights[ 1 ]- bsplineWeights[ 2 ];
  } // end GetWeights()

  static inline void GetDerivativeWeights( itk::Vector<double,3> & bsplineWeightsD,
    const TCoefficientType & w )
  {
    TCoefficientType wr = 1.0 - w;

    bsplineWeightsD[ 0 ] = 0.0 - wr;
    bsplineWeightsD[ 1 ] = wr - w;
    bsplineWeightsD[ 2 ] = w;
  } // end GetDerivativeWeights()

  //    static inline void
  //    getHessianWeights( itk::Vector<double,3> &bsplweightsH,
  //                       const TCoefficientType & w )
  //    {
  //        bsplweightsH[0] = 1.0; //To be implemented
  //        bsplweightsH[1] = -2.0; //To be implemented
  //        bsplweightsH[2] = 1.0; //To be implemented
  //    }
};

// Specialization for B-spline order 3: Cubic interpolation
template< class TCoefficientType >
class BSplineWeights< 3, TCoefficientType >
{
public:
  static inline void GetWeights( itk::Vector<double,4> & bsplineWeights,
    const TCoefficientType & w )
  {
    TCoefficientType sqr_w  = w * w;

    bsplineWeights[ 3 ] = (1.0 / 6.0) * sqr_w * w;
    bsplineWeights[ 0 ] = (1.0 / 6.0) + 0.5 * w * ( w - 1.0 ) - bsplineWeights[3];
    bsplineWeights[ 2 ] = w + bsplineWeights[0] - 2.0 * bsplineWeights[3];
    bsplineWeights[ 1 ] = 1.0 - bsplineWeights[0] - bsplineWeights[2] - bsplineWeights[3];
  } // end GetWeights()

  static inline void GetDerivativeWeights( itk::Vector<double,4> & bsplineWeightsD,
    const TCoefficientType & w )
  {
    TCoefficientType w1, w2, w3;
    w2 = .75 - w * w;
    w3 = 0.5 * ( w - w2 + 1.0 );
    w1 = 1.0 - w2 - w3;

    bsplineWeightsD[ 0 ] = 0.0 - w1;
    bsplineWeightsD[ 1 ] = w1 - w2;
    bsplineWeightsD[ 2 ] = w2 - w3;
    bsplineWeightsD[ 3 ] = w3;
  } // end GetDerivativeWeights()

  //    static inline void
  //    getHessianWeights( itk::Vector<double,4> &bsplweightsH,
  //                       const TCoefficientType & w )
  //    {
  //        bsplweightsH[0] = 1.0-w; //To be implemented
  //        bsplweightsH[1] = 3.0*w-2.0; //To be implemented
  //        bsplweightsH[2] = 1.0-3*w; //To be implemented
  //        bsplweightsH[3] = w; //To be implemented
  //    }
};

// Specialization for B-spline order 4
template< class TCoefficientType >
class BSplineWeights< 4, TCoefficientType >
{
public:
  static inline void GetWeights( itk::Vector<double,5> & bsplineWeights,
    const TCoefficientType & w )
  {
    TCoefficientType w_sqr, t, t0, t1;

    w_sqr = w * w;
    t = ( 1.0 / 6.0 ) * w_sqr;
    bsplineWeights[0] = 0.5 - w;
    bsplineWeights[0] *= bsplineWeights[0];
    bsplineWeights[0] *= ( 1.0 / 24.0 ) * bsplineWeights[0];

    t0 = w * ( t - 11.0 / 24.0 );
    t1 = 19.0 / 96.0 + w_sqr * ( 0.25 - t );

    bsplineWeights[1] = t1 + t0;
    bsplineWeights[3] = t1 - t0;
    bsplineWeights[4] = bsplineWeights[0] + t0 + 0.5 * w;
    bsplineWeights[2] = 1.0 - bsplineWeights[0] - bsplineWeights[1] - bsplineWeights[3] - bsplineWeights[4];
  } // end GetWeights()

  static inline void GetDerivativeWeights( itk::Vector<double,5> & bsplineWeightsD,
    const TCoefficientType & w )
  {
    TCoefficientType w1, w2, w3, w4;
    w4 = (1.0 / 6.0 ) * w * w * w;
    w1 = (1.0 / 6.0 ) + 0.5 * w * ( w - 1.0 ) - w4;
    w3 = w + w1 - 2.0 * w4;
    w2 = 1.0 - w1 - w3 - w4;

    bsplineWeightsD[0] = 0.0 - w1;
    bsplineWeightsD[1] = w1 - w2;
    bsplineWeightsD[2] = w2 - w3;
    bsplineWeightsD[3] = w3 - w4;
    bsplineWeightsD[4] = w4;
  } // end GetDerivativeWeights()

  //    static inline void
  //    getHessianWeights( itk::Vector<double,5> &bsplweightsH,
  //                       const TCoefficientType & w )
  //    {
  //        TCoefficientType w_sqr = w*w;

  //        bsplweightsH[0] = 1/2*w_sqr - 9/2*w + 81/8;
  //        bsplweightsH[1] = -1/2*(4*w_sqr - 18*w + 19);
  //        bsplweightsH[2] = 3*w_sqr - 5/4;
  //        bsplweightsH[3] = -1/2*(4*w_sqr + 18*w + 19);
  //        bsplweightsH[4] = 1/2*w_sqr + 9/2*w + 81/8;

  //    }
};

// Specialization for B-spline order 5
template< class TCoefficientType >
class BSplineWeights< 5, TCoefficientType >
{
public:
  static inline void GetWeights( itk::Vector<double,6> & bsplineWeights,
    const TCoefficientType & w )
  {
    TCoefficientType w_sqr, w_qua, w2, t, t0, t1;
    w_sqr = w * w;
    bsplineWeights[5] = ( 1.0 / 120.0 ) * w * w_sqr * w_sqr;

    w_sqr -= w;
    w_qua = w_sqr * w_sqr;
    w2 = w-0.5;
    t = w_sqr * (w_sqr - 3.0 );

    bsplineWeights[0] = ( 1.0 / 24.0 ) * ( 1.0 / 5.0 + w_sqr + w_qua ) - bsplineWeights[5];

    t0 = (1.0 / 24.0 ) * ( w_sqr * (w_sqr - 5.0 ) + 46.0 / 5.0 );
    t1 = ( -1.0 / 12.0 ) * w2 * ( t + 4.0 );

    bsplineWeights[2] = t0 + t1;
    bsplineWeights[3] = t0 - t1;

    t0 = ( 1.0 / 16.0 ) * ( 9.0 / 5.0 - t );
    t1 = ( 1.0 / 24.0 ) * w2 * ( w_qua - w_sqr - 5.0 );

    bsplineWeights[1] = t0 + t1;
    bsplineWeights[4] = t0 - t1;
  } // end GetWeights()

  static inline void GetDerivativeWeights( itk::Vector<double,6> & bsplineWeightsD,
    const TCoefficientType & w )
  {
    TCoefficientType w_sqr, t, t0, t1, w1, w2, w3, w4, w5;

    w_sqr = w * w;
    t = (1.0 / 6.0 ) * w_sqr;
    w1 = 0.5 - w;
    w1 *= w1;
    w1 *= ( 1.0 / 24.0 ) * w1;
    t0 = w * ( t - 11.0 / 24.0 );
    t1 = 19.0 / 96.0 + w_sqr * ( 0.25 - t );
    w2 = t1 + t0;
    w4 = t1 - t0;
    w5 = w1 + t0 + 0.5 * w;
    w3 = 1.0 - w1 - w2 - w4 - w5;

    bsplineWeightsD[0] = 0.0 - w1;
    bsplineWeightsD[1] = w1 - w2;
    bsplineWeightsD[2] = w2 - w3;
    bsplineWeightsD[3] = w3 - w4;
    bsplineWeightsD[4] = w4 - w5;
    bsplineWeightsD[5] = w5;
  } // end GetDerivativeWeights()

  //    static inline void
  //    getHessianWeights( itk::Vector<double,6> &bsplweightsH,
  //                       const TCoefficientType & w )
  //    {
  //        bsplweightsH[0] = 1; //To be implemented
  //        bsplweightsH[1] = 1; //To be implemented
  //        bsplweightsH[2] = 1; //To be implemented
  //        bsplweightsH[3] = 1; //To be implemented
  //        bsplweightsH[4] = 1; //To be implemented
  //        bsplweightsH[5] = 1; //To be implemented
  //    }
};

} // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRecursiveBSplineInterpolateImageFunction.hxx"
#endif

#endif //__itkRecursiveBSplineInterpolateImageFunction_h
