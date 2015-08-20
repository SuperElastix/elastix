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
#ifndef __itkRecursiveBSplineImplementation_h
#define __itkRecursiveBSplineImplementation_h

// temporary preprocessor directive to strongly request inlining.
// This request is warranted as the functions benefit quite strongly from inlining
// and each function should be only called once with the same template arguments
// by a wrapping function; so there is no space benefit of not inlining.
#ifdef _MSC_VER
  #define FORCEINLINE __forceinline
#else
  #define FORCEINLINE inline
#endif

namespace itk
{
/** \class RecursiveBSplineImplementation
 *
 * \brief This set of helper classes contains the actual implementation of the
 * recursive B-spline transform
 *
 * Compared to the RecursiveBSplineTransformImplementation class, this
 * class is templated over the OutputType and input pointer type and the
 * interpolated point is returned by value.
 * This allows using efficient (SSE/AVX optimized) vector implementation for
 * GetSample. Also this same version can now be used to sample scalar as
 * well as vector images (of compile time known vector length, although with the
 * right selection of types also dynamic vector lengths; but that will nececarily
 * be less efficient.)
 *
 * ASSUMPTIONS:
 *   The optimized end cases (may) assume that gridOffsetTable[0] == 1 (element increment in InputPointerType)
 *
 * STRUCTURE:
 *   each method has it's own class with only 1 method:
 *      RecursiveBSplineImplementation_(methodName)::(methodName)( ... )
 *   where '(methodName)' is one of the following:
 *      GetSample                  // old name: TransformPoint
 *      GetSpatialJacobian
 *      GetSpatialHessian
 *      GetJacobian                // Note that it is almost always more efficient to use MultiplyJacobianWithValue.
 *      ComputeNonZeroJacobianIndices
 *      MultiplyJacobianWithValue  // similar to old name, which I consider too specific: EvaluateJacobianWithImageGradientProduct
 *                                 // This is the adjoint of 'GetSample' :
 *                                 //    if you regard the sampling as matrix multiplication: GetSample(mu, ..) == Jacobian * mu
 *                                 //    then MultiplyJacobianWithValue( jac, value, ...) == transpose(Jacobian) * value
 *   REASONS:
 *   - To allow easy addition of specialized end cases.
 *   - To keep the code of each method, including the end cases, together in the file.
 *   - To allow variations in template arguments.
 *   - The reason to put a static method in a class rather than using functions is that
 *     c++ does not allow partial specialization of function templates, which in this
 *     case makes it impossible to construct the end cases.
 *
 * Other remarks:
 *   - I (DPoot) have exerimented with prefetching. However, at this moment my conclusion is that that does not
 *     really help when the images are large, and seriously hurts performance for small images. I think the main
 *     reason is that the prefetching adds so many instructions that latency cannot be hided as efficiently. Also
 *     the (effective) instruction throughput seriously reduces as without prefetching (and small images) already
 *     the maximum throughput is almost reached.
 *   - If you need to sample vector images, use an (SSE/AVX optimized) fixed-length vector class for OutputType and a compatible InputPointerType
 *     for example vec< type, vlen> with the accompagnying vecptr< type, vlen> as InputPointerType.
 *
 * \ingroup ITKTransform
 */

#define USE_STEPS 2873462 // USE_STEPS is used as gridOffsetTable0 value. If set a 'steps' argument is assumed, instead of the 'gridOffsetTable' argument.
                          // Function overloading to accomplish this is not possible, since both have the same type. Hence we need a template argument
                          // for differntiating this.
                          // The main reason to reuse the 'gridOffsetTable0' argument is that this already changes the interpretation of the gridOffsetTable argument.
                          // Use a large integer for USE_STEPS to make sure that if it (accidently) is interpreted as gridOffsetTable0 it will most likely cause
                          // segfaults (usefull to diagnose the issue). Also it is extremely unlikely that a actual gridOffsetTable0 is such a large integer.



template< class OutputType, unsigned int SpaceDimension, unsigned int SplineOrder, class InputPointerType, int gridOffsetTable0 = 1 >
class RecursiveBSplineImplementation_GetSample
{
public:
  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** GetSample recursive implementation. */
  static FORCEINLINE OutputType GetSample(
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D )
  {
    /** Make a copy of the pointers to mu. The pointer will move later. */
    InputPointerType tmp_mu = mu;
    /*InputPointerType tmp_prefetch_mu;
    if (doPrefetch)
      tmp_prefetch_mu = prefetch_mu;*/

    /** Create a temporary sj and initialize the original. */
    OutputType opp( 0.0 );

    const OffsetValueType bot = SpaceDimension == 1 && gridOffsetTable0 != 0
      ? gridOffsetTable0 : gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      if (gridOffsetTable0==USE_STEPS)
        tmp_mu = mu + gridOffsetTable[k + HelperConstVariable];
      opp += RecursiveBSplineImplementation_GetSample< OutputType, SpaceDimension - 1, SplineOrder, InputPointerType, gridOffsetTable0 >
        ::GetSample( tmp_mu, gridOffsetTable, weights1D ) * weights1D[ k + HelperConstVariable ];
      //::GetSample2( tmp_mu, gridOffsetTable, weights1D , tmp_prefetch_mu ) * weights1D[ k + HelperConstVariable ];

      // move to the next mu
      if (gridOffsetTable0!=USE_STEPS)
        tmp_mu += bot;
      /*if (doPrefetch)
        tmp_prefetch_mu +=bot;*/
    }
    return opp;
  } // end GetSample()
}; // end class


/** \class RecursiveBSplineImplementation_GetSample
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< class OutputType, unsigned int SplineOrder, class InputPointerType, int gridOffsetTable0 >
class RecursiveBSplineImplementation_GetSample< OutputType, 0, SplineOrder, InputPointerType, gridOffsetTable0 >
{
public:
  /** GetSample recursive implementation. */
  static FORCEINLINE  OutputType GetSample(
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D ) //, const InputPointerType prefetch_mu)
  {
   /*if (doPrefetch)
     _mm_prefetch( (const char *) prefetch_mu , _MM_HINT_T2 ) ;*/
     return *mu;
  } // end GetSample()

}; // end class


/** \class RecursiveBSplineImplementation_GetSample
 *
 * \brief Define the specialized end case for SpaceDimension = 1 and SplineOrder == 3.
 */

template< class OutputType, class InputPointerType, int gridOffsetTable0 >
class RecursiveBSplineImplementation_GetSample< OutputType, 1, 3, InputPointerType, gridOffsetTable0 >
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  static const unsigned int SpaceDimension = 1;
  static const unsigned int SplineOrder = 3;
  //static const unsigned int CACHE_LINE_SIZE = 64;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** GetSample recursive implementation. */
  static FORCEINLINE  OutputType GetSample(
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D ) // , const InputPointerType prefetch_mu)
  {
    OutputType weights0( weights1D[ 0 + HelperConstVariable ] );
    OutputType weights1( weights1D[ 1 + HelperConstVariable ] );
    OutputType weights2( weights1D[ 2 + HelperConstVariable ] );
    OutputType weights3( weights1D[ 3 + HelperConstVariable ] );
    /*if (doPrefetch) {
      _mm_prefetch( ( (const char *) prefetch_mu ) + 0 * CACHE_LINE_SIZE, _MM_HINT_T2 ) ; // first byte used.
      _mm_prefetch( ( (const char *) prefetch_mu ) + 1 * CACHE_LINE_SIZE, _MM_HINT_T2 ) ;
      //if ( ( (const char *) prefetch_mu ) + 2 * CACHE_LINE_SIZE < lastByteUsed) // we need some compile time test to determine how many cache lines to prefetch.
      //_mm_prefetch( ( (const char *) prefetch_mu ) + 2 * CACHE_LINE_SIZE, _MM_HINT_T2 ) ;
      const char * lastByteUsed = ( (const char *) (prefetch_mu+4) ) - 1;
      _mm_prefetch( lastByteUsed, _MM_HINT_T2 ) ; // last byte used.
    }*/
    if (gridOffsetTable0==USE_STEPS) {
      return (   (*(mu + gridOffsetTable[0 + HelperConstVariable])) * weights0
               + (*(mu + gridOffsetTable[1 + HelperConstVariable])) * weights1 )
            +(   (*(mu + gridOffsetTable[2 + HelperConstVariable])) * weights2
               + (*(mu + gridOffsetTable[3 + HelperConstVariable])) * weights3 ) ;
    } else {
      const OffsetValueType bot = SpaceDimension == 1 && gridOffsetTable0 != 0
        ? gridOffsetTable0 : gridOffsetTable[ SpaceDimension - 1 ];
      return ( (*(mu+0)) * weights0 + (*(mu + 1 * bot))  * weights1  )
        + ( (*(mu + 2 * bot)) * weights2 + (*(mu + 3 * bot))  * weights3 );
    }
  } // end GetSample()

}; // end class

/*  Surprisingly, the very specialized case below is a bit slower than the version above.

template< bool doPrefetch > class RecursiveBSplineTransformImplementation4< vec<double, 3>, 1, 3,  vecptr< double * , 3> , doPrefetch>
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   * /
  static const unsigned int SpaceDimension = 1;
  static const unsigned int SplineOrder =3;
  typedef vec<double, 3> OutputType ;
  typedef vecptr<  double * , 3> InputPointerType;

  /** Helper constant variable. * /
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** GetSample recursive implementation. * /
  static inline OutputType GetSample2(
    const InputPointerType vecmu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D )
  {
    typedef __m128d vT;
  const double * mu = vecmu.getRawPointer();

  vT weights01 = _mm_loadu_pd( & weights1D[ 0 + HelperConstVariable ] );
  vT weights23 = _mm_loadu_pd( & weights1D[ 2 + HelperConstVariable ] );

  vT mu01 = _mm_loadu_pd( mu     );
  vT mu23 = _mm_loadu_pd( mu + 2 );
  vT mu45 = _mm_loadu_pd( mu + 4 );
  vT mu67 = _mm_loadu_pd( mu + 6 );
  vT mu89 = _mm_loadu_pd( mu + 8 );
  vT muAB = _mm_loadu_pd( mu +10 );

  vT res20 = _mm_add_pd( _mm_mul_pd(weights01,mu23) , _mm_mul_pd(weights23,mu89) );
  vT weights0 = _mm_unpacklo_pd(weights01, weights01);
  vT weights1 = _mm_unpackhi_pd(weights01, weights01);
  vT weights2 = _mm_unpacklo_pd(weights23, weights23);
  vT weights3 = _mm_unpackhi_pd(weights23, weights23);
  vT res01    = _mm_add_pd( _mm_mul_pd( weights0 , mu01), _mm_mul_pd( weights2, mu67 ) );
  vT res12    = _mm_add_pd( _mm_mul_pd( weights1 , mu45), _mm_mul_pd( weights3, muAB ) );
  res01 = _mm_add_pd( res01, _mm_shuffle_pd( res20, res12, _MM_SHUFFLE2(0, 1) ) );
  vT res2 =  _mm_add_sd( res20, _mm_unpackhi_pd( res12, res12 ) ) ;
  return vec< double, 3>( res01, res2 );
  } // end GetSample()



}; // end class
*/


/** \class RecursiveBSplineImplementation_GetSpatialJacobian
 *
 * \brief Define general case
 */

template< class OutputPointerType, unsigned int SpaceDimension, unsigned int SplineOrder, class InputPointerType, int gridOffsetTable0 = 1 >
class RecursiveBSplineImplementation_GetSpatialJacobian
{
public:
  typedef typename std::iterator_traits< OutputPointerType >::value_type OutputValueType; //\todo: is this the proper use of std::iterator_traits? Preferably we use 'using std::iterator_traits', to allow custom template specializations.
  typedef OutputValueType * RecursiveOutputPointerType;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** GetSpatialJacobian recursive implementation. */
  static inline void GetSpatialJacobian(
    OutputPointerType sj,
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D )
  {
    /** Make a copy of the pointers to mu. The pointer will move later. */
    InputPointerType tmp_mu = mu;

    /** Create a temporary sj and initialize the original. */
    OutputValueType  tmp_sj[ SpaceDimension ];
    for( unsigned int n = 0; n < SpaceDimension + 1 ; ++n )
    {
      sj[ n ] = 0.0;
    }

    const OffsetValueType bot = SpaceDimension == 1 && gridOffsetTable0 != 0
      ? gridOffsetTable0 : gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      if (gridOffsetTable0==USE_STEPS)
        tmp_mu = mu + gridOffsetTable[k + HelperConstVariable];
      RecursiveBSplineImplementation_GetSpatialJacobian< RecursiveOutputPointerType, SpaceDimension - 1, SplineOrder, InputPointerType, gridOffsetTable0 >
        ::GetSpatialJacobian( tmp_sj, tmp_mu, gridOffsetTable, weights1D , derivativeWeights1D );

      // Multiply by the weights
      for( unsigned int n = 0; n < SpaceDimension; ++n )
      {
        sj[ n ] += tmp_sj[ n ] * weights1D[ k + HelperConstVariable ];
      }

      // Multiply by the derivative weights
      sj[ SpaceDimension ] += tmp_sj[ 0 ] * derivativeWeights1D[ k + HelperConstVariable ];

      // move to the next mu
      if (gridOffsetTable0!=USE_STEPS)
        tmp_mu += bot;
    }
  } // end GetSpatialJacobian()
}; // end class


/** \class RecursiveBSplineImplementation_GetSpatialJacobian
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< class OutputPointerType, unsigned int SplineOrder, class InputPointerType, int gridOffsetTable0 >
class RecursiveBSplineImplementation_GetSpatialJacobian< OutputPointerType, 0, SplineOrder, InputPointerType, gridOffsetTable0 >
{
public:
  /** GetSpatialJacobian recursive implementation. */
  static inline void GetSpatialJacobian(
    OutputPointerType sj,
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D )
  {
    *sj = *mu;
  } // end GetSpatialJacobian()
}; // end class


/** \class RecursiveBSplineImplementation_GetSpatialJacobian
 *
 * \brief Define the specialized end case for SpaceDimension = 1 and SplineOrder == 3.
 */

template< class OutputPointerType, class InputPointerType, int gridOffsetTable0 >
class RecursiveBSplineImplementation_GetSpatialJacobian< OutputPointerType, 1, 3,  InputPointerType, gridOffsetTable0 >
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  typedef typename std::iterator_traits< OutputPointerType >::value_type OutputValueType; //\todo: is this the proper use of std::iterator_traits? Preferably we use 'using std::iterator_traits', to allow custom template specializations.
  static const unsigned int SpaceDimension = 1;
  static const unsigned int SplineOrder = 3;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** GetSpatialJacobian recursive implementation. */
  static inline void GetSpatialJacobian(
    OutputPointerType sj,
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D )
  {
    const OffsetValueType bot = SpaceDimension == 1 && gridOffsetTable0 != 0
      ? gridOffsetTable0 : gridOffsetTable[ SpaceDimension - 1 ];
    OutputValueType mu0 = *( gridOffsetTable0==USE_STEPS ? mu + gridOffsetTable[0 + HelperConstVariable] : mu + 0 * bot);
    OutputValueType mu1 = *( gridOffsetTable0==USE_STEPS ? mu + gridOffsetTable[1 + HelperConstVariable] : mu + 1 * bot);
    OutputValueType mu2 = *( gridOffsetTable0==USE_STEPS ? mu + gridOffsetTable[2 + HelperConstVariable] : mu + 2 * bot);
    OutputValueType mu3 = *( gridOffsetTable0==USE_STEPS ? mu + gridOffsetTable[3 + HelperConstVariable] : mu + 3 * bot);

    sj[ 0 ] = ( mu0 * weights1D[ 0 + HelperConstVariable ] + mu1 * weights1D[ 1 + HelperConstVariable ] )
            + ( mu2 * weights1D[ 2 + HelperConstVariable ] + mu3 * weights1D[ 3 + HelperConstVariable ] );
    sj[ 1 ] = ( mu0 * derivativeWeights1D[ 0 + HelperConstVariable ] + mu1 * derivativeWeights1D[ 1 + HelperConstVariable ] )
            + ( mu2 * derivativeWeights1D[ 2 + HelperConstVariable ] + mu3 * derivativeWeights1D[ 3 + HelperConstVariable ] );
  } // end GetSpatialJacobian()

}; // end class


/** \class RecursiveBSplineImplementation_GetSpatialHessian
 *
 * \brief Define general case
 */

template< class OutputPointerType, unsigned int SpaceDimension, unsigned int SplineOrder, class InputPointerType, int gridOffsetTable0 = 1 >
class RecursiveBSplineImplementation_GetSpatialHessian
{
public:
  typedef typename std::iterator_traits< OutputPointerType >::value_type OutputValueType; //\todo: is this the proper use of std::iterator_traits? Preferably we use 'using std::iterator_traits', to allow custom template specializations.
  typedef OutputValueType * RecursiveOutputPointerType;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** GetSpatialHessian recursive implementation. */
  static inline void GetSpatialHessian(
    OutputPointerType sh,
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D )   // 2nd derivative of B-spline
  {
    const unsigned int helperDim1 = SpaceDimension * ( SpaceDimension + 1 ) / 2;
    const unsigned int helperDim2 = ( SpaceDimension + 1 ) * ( SpaceDimension + 2 ) / 2;

    /** Make a copy of the pointers to mu. The pointer will move later. */
    InputPointerType tmp_mu = mu;

    /** Create a temporary sh and initialize the original. */
    OutputValueType  tmp_sh[ helperDim1 ];
    for( unsigned int n = 0; n < helperDim2 ; ++n )
    {
      sh[ n ] = 0.0;
    }

    const OffsetValueType bot = SpaceDimension == 1 && gridOffsetTable0 != 0
      ? gridOffsetTable0 : gridOffsetTable[ SpaceDimension - 1 ];
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      if (gridOffsetTable0==USE_STEPS)
        tmp_mu = mu + gridOffsetTable[k + HelperConstVariable];
      RecursiveBSplineImplementation_GetSpatialHessian< RecursiveOutputPointerType, SpaceDimension - 1, SplineOrder, InputPointerType, gridOffsetTable0 >
        ::GetSpatialHessian( tmp_sh, tmp_mu, gridOffsetTable, weights1D, derivativeWeights1D, hessianWeights1D );

      // Multiply by the weights
      for( unsigned int n = 0; n < helperDim1; ++n )
      {
        sh[ n ] += tmp_sh[ n ] * weights1D[ k + HelperConstVariable ];
      }

      // Multiply by the derivative weights
      for( unsigned int n = 0; n < SpaceDimension; ++n )
      {
        sh[ n + helperDim1 ] += tmp_sh[ n * ( n + 1 ) / 2 ] * derivativeWeights1D[ k + HelperConstVariable ];
      }

      // Multiply by the Hessian weights
      sh[ helperDim2 - 1 ] += tmp_sh[ 0 ] * hessianWeights1D[ k + HelperConstVariable ];

      // move to the next mu
      if (gridOffsetTable0!=USE_STEPS)
        tmp_mu += bot;
    }
  } // end GetSpatialHessian()
}; // end class


/** \class RecursiveBSplineImplementation_GetSpatialHessian
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< class OutputPointerType, unsigned int SplineOrder, class InputPointerType, int gridOffsetTable0 >
class RecursiveBSplineImplementation_GetSpatialHessian< OutputPointerType, 0, SplineOrder, InputPointerType, gridOffsetTable0 >
{
public:
  /** GetSpatialHessian recursive implementation. */
  static inline void GetSpatialHessian(
    OutputPointerType sh,
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D )   // 2nd derivative of B-spline
  {
    *sh = *mu;
  } // end GetSpatialHessian()
}; // end class


/** \class RecursiveBSplineImplementation_GetSpatialHessian
 *
 * \brief Define the end case for SpaceDimension = 1 for SplineOrder == 3 to improve speed for this common SplineOrder case.
 */

template< class OutputPointerType, class InputPointerType, int gridOffsetTable0 >
class RecursiveBSplineImplementation_GetSpatialHessian< OutputPointerType, 1, 3, InputPointerType, gridOffsetTable0 >
{
public:
  /** Typedef related to the coordinate representation type and the weights type.
   * Usually double, but can be float as well. <Not tested very well for float>
   */
  typedef typename std::iterator_traits< OutputPointerType >::value_type OutputValueType; //\todo: is this the proper use of std::iterator_traits? Preferably we use 'using std::iterator_traits', to allow custom template specializations.
  static const unsigned int SpaceDimension = 1;
  static const unsigned int SplineOrder = 3;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** GetSpatialHessian recursive implementation. */
  static inline void GetSpatialHessian(
    OutputPointerType sh,
    const InputPointerType mu,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D,
    const double * derivativeWeights1D, // 1st derivative of B-spline
    const double * hessianWeights1D )   // 2nd derivative of B-spline
  {
    const unsigned int helperDim1 = SpaceDimension * ( SpaceDimension + 1 ) / 2;  // == 1
    const unsigned int helperDim2 = ( SpaceDimension + 1 ) * ( SpaceDimension + 2 ) / 2; //==3

    const OffsetValueType bot = SpaceDimension == 1 && gridOffsetTable0 != 0
      ? gridOffsetTable0 : gridOffsetTable[ SpaceDimension - 1 ];
    OutputValueType mu0 = *( gridOffsetTable0==USE_STEPS ? mu + gridOffsetTable[0 + HelperConstVariable] : mu + 0 * bot);
    OutputValueType mu1 = *( gridOffsetTable0==USE_STEPS ? mu + gridOffsetTable[1 + HelperConstVariable] : mu + 1 * bot);
    OutputValueType mu2 = *( gridOffsetTable0==USE_STEPS ? mu + gridOffsetTable[2 + HelperConstVariable] : mu + 2 * bot);
    OutputValueType mu3 = *( gridOffsetTable0==USE_STEPS ? mu + gridOffsetTable[3 + HelperConstVariable] : mu + 3 * bot);

    sh[ 0 ] = ( mu0 * weights1D[ 0 + HelperConstVariable ] + mu1 * weights1D[ 1 + HelperConstVariable ] )
            + ( mu2 * weights1D[ 2 + HelperConstVariable ] + mu3 * weights1D[ 3 + HelperConstVariable ] );
    sh[ 1 ] = ( mu0 * derivativeWeights1D[ 0 + HelperConstVariable ] + mu1 * derivativeWeights1D[ 1 + HelperConstVariable ] )
            + ( mu2 * derivativeWeights1D[ 2 + HelperConstVariable ] + mu3 * derivativeWeights1D[ 3 + HelperConstVariable ] );
    sh[ 2 ] = ( mu0 * hessianWeights1D[ 0 + HelperConstVariable ] + mu1 * hessianWeights1D[ 1 + HelperConstVariable ] )
            + ( mu2 * hessianWeights1D[ 2 + HelperConstVariable ] + mu3 * hessianWeights1D[ 3 + HelperConstVariable ] );
  } // end GetSpatialHessian()

}; // end class


/** \class RecursiveBSplineImplementation_numberOfPointsInSupportRegion
 *
 * \brief Define general case
 */
template< unsigned int SpaceDimension, unsigned int SplineOrder >
class RecursiveBSplineImplementation_numberOfPointsInSupportRegion
{
public:
  /** Helper constant variable. */
  //itkStaticConstMacro( numberOfPointsInSupportRegion, unsigned int,
  //  ( SplineOrder + 1 ) * RecursiveBSplineImplementation_numberOfPointsInSupportRegion< SpaceDimension-1, SplineOrder>::numberOfPointsInSupportRegion );
  enum { numberOfPointsInSupportRegion = ( SplineOrder + 1 ) * RecursiveBSplineImplementation_numberOfPointsInSupportRegion< SpaceDimension-1, SplineOrder>::numberOfPointsInSupportRegion };
}; // end class
//End case:
template<unsigned int SplineOrder>
class RecursiveBSplineImplementation_numberOfPointsInSupportRegion< 0, SplineOrder >
{
  public:
  /** Helper constant variable. */
  // itkStaticConstMacro( numberOfPointsInSupportRegion, unsigned int, 1 );
    enum { numberOfPointsInSupportRegion = 1 };
};

/** \class RecursiveBSplineImplementation_GetJacobian
 *
 * \brief Define general case
 */

template< class OutputPointerType, unsigned int SpaceDimension, unsigned int SplineOrder, class InputValueType >
class RecursiveBSplineImplementation_GetJacobian
{
public:
  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** GetJacobian recursive implementation. */
  static FORCEINLINE void GetJacobian(
    OutputPointerType & jacobians,
    const double * weights1D,
    InputValueType value )
  {
    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      RecursiveBSplineImplementation_GetJacobian< OutputPointerType, SpaceDimension - 1, SplineOrder, InputValueType >
        ::GetJacobian( jacobians, weights1D, value * weights1D[ k + HelperConstVariable ] ) ;
    }
  } // end GetJacobian()
}; // end class


/** \class RecursiveBSplineImplementation_GetJacobian
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< class OutputPointerType, unsigned int SplineOrder, class InputValueType >
class RecursiveBSplineImplementation_GetJacobian< OutputPointerType, 0, SplineOrder, InputValueType >
{
public:
  /** getJacobian recursive implementation. */
  static FORCEINLINE void GetJacobian(
    OutputPointerType & jacobians,
    const double * weights1D,
    InputValueType value )
  {
    *jacobians = value;
    ++jacobians;
  } // end GetJacobian()
}; // end class


/** \class RecursiveBSplineImplementation_ComputeNonZeroJacobianIndices
 *
 * \brief Define general case
 * This class computes the nonzero Jacobian indices (which also are the nonzero Jacobian of SpatialJacobian and Jacobian of SpatialHessian indices)
 * This class is almost never needed. Typically use MultiplyJacobianWithValue as that implicitly computes these indices.
 *
 * Special remark:
 *  if GetJacobian is called with a vector OutputPointerType, ComputeNonZeroJacobianIndices should be called with
 *  this vector dimension as first dimension.
 * e.g.:
 *    RecursiveBSplineImplementation_GetJacobian< vecptr< double *, vecLength>, SpatialDimension, SplineOrder, double>
 *      ::getJacobian( jacobians, weights1D, value )
 *    // note the typical call has vecLength == SpatialDimension
 * has as matching call:
 *    scaledGridOffsetTable[ SpatialDimension ]
 *    for (i = 1; i < SpatialDimension; ++i ) {
 *       scaledGridOffsetTable[i] = gridOffsetTable[i]*vecLength;
 *    }
 *    temp_nzji = nzji;
 *    CurrentIndexArray[vecLength];
 *    for (int i = 0; i < vecLength; ++i ) {
 *      CurrentIndexArray[i] = CurrentIndex+i;
 *    }
 *    vec< int, vecLength> vecCurrentIndex( & CurrentIndexArray[0] )
 *    RecursiveBSplineImplementation_ComputeNonZeroJacobianIndices< vecptr< int *, vecLength>, SpatialDimension , SplineOrder, vec< int, vecLength> >
 *      ::ComputeNonZeroJacobianIndices( temp_nzji, vecCurrentIndex, &scaledGridOffsetTable[0] );
 *
 * Note: the input argument nzji is incremented to the end.
 */

template< class OutputPointerType, unsigned int SpaceDimension, unsigned int SplineOrder, class InputValueType, int gridOffsetTable0 = 1 >
class RecursiveBSplineImplementation_ComputeNonZeroJacobianIndices
{
public:
  typedef typename std::iterator_traits< OutputPointerType >::value_type indexType;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** ComputeNonZeroJacobianIndices recursive implementation. */
  static inline void ComputeNonZeroJacobianIndices(
    OutputPointerType & nzji,
    InputValueType currentIndex,
    const OffsetValueType * gridOffsetTable)
  {
    InputValueType tmp_currentIndex = currentIndex;
    const OffsetValueType bot = SpaceDimension == 1 && gridOffsetTable0 != 0
      ? gridOffsetTable0 : gridOffsetTable[ SpaceDimension - 1 ];

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      if (gridOffsetTable0==USE_STEPS)
        tmp_currentIndex = currentIndex + gridOffsetTable[k + HelperConstVariable];
      RecursiveBSplineImplementation_ComputeNonZeroJacobianIndices< OutputPointerType, SpaceDimension - 1, SplineOrder, InputValueType, gridOffsetTable0 >
        ::ComputeNonZeroJacobianIndices( nzji, tmp_currentIndex , gridOffsetTable );
      if (gridOffsetTable0!=USE_STEPS)
        tmp_currentIndex  += bot;
    }
  } // end ComputeNonZeroJacobianIndices()
}; // end class


/** \class RecursiveBSplineImplementation_ComputeNonZeroJacobianIndices
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< class OutputPointerType, unsigned int SplineOrder, class InputValueType, int gridOffsetTable0 >
class RecursiveBSplineImplementation_ComputeNonZeroJacobianIndices< OutputPointerType, 0, SplineOrder, InputValueType, gridOffsetTable0 >
{
public:
  typedef typename std::iterator_traits< OutputPointerType >::value_type indexType;

  /** ComputeNonZeroJacobianIndices recursive implementation. */
  static inline void ComputeNonZeroJacobianIndices(
    OutputPointerType & nzji,
    InputValueType currentIndex,
    const OffsetValueType * gridOffsetTable )
  {
    *nzji = currentIndex;
    ++nzji;
  } // end ComputeNonZeroJacobianIndices()
}; // end class


/** \class RecursiveBSplineImplementation_MultiplyJacobianWithValue
 *
 * \brief Define general case
 * This class should almost always be preferred used over GetSpatialJacobian and ComputeNonZeroJacobianIndices.
 *
 * INPUTS:
 *    gradCoefficients : identical type (and size) to mu input in the corresponding GetSample.
 *                       The result of the multiplication with the Jacobian is added to this vector.
 *    weight : scaling applied to multiplication (typically == 1.0)
 *    gradValue : same type as output of GetSample. Typically contains the image gradient.
 *                You might want to add a reference in InputValueType (so that gradValue is passed by reference, rather than by value)
 *    gridOffsetTable & weights1D : identical to the ones used in GetSample.
 *
 * NOTE for programmers of this class:
 *   Currently there is a difference in recursion between GetSample and MultiplyJacobianWithValue.
 *   This difference is not needed; in MultiplyJacobianWithValue we could multiply gradValue by weights1D[..] or, alternatively
 *   we could pass a 'weight' value also in GetSample. Currently do not know which recursion is more efficient.
 */

template< class gradPointerType, unsigned int SpaceDimension, unsigned int SplineOrder, class InputValueType, int gridOffsetTable0 = 1 >
class RecursiveBSplineImplementation_MultiplyJacobianWithValue
{
public:
  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** MultiplyJacobianWithValue recursive implementation. */
  static FORCEINLINE void MultiplyJacobianWithValue(
    gradPointerType gradCoefficients,
    double weight,
    const InputValueType gradValue,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D )
  {
    gradPointerType tmp_gradCoefficients = gradCoefficients;
    const OffsetValueType bot = SpaceDimension == 1 && gridOffsetTable0 != 0
      ? gridOffsetTable0 : gridOffsetTable[ SpaceDimension - 1 ];

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      if (gridOffsetTable0==USE_STEPS)
        tmp_gradCoefficients = gradCoefficients + gridOffsetTable[k + HelperConstVariable];
      RecursiveBSplineImplementation_MultiplyJacobianWithValue< gradPointerType, SpaceDimension - 1, SplineOrder, InputValueType, gridOffsetTable0 >
        ::MultiplyJacobianWithValue( tmp_gradCoefficients, weight * weights1D[ k + HelperConstVariable ], gradValue , gridOffsetTable, weights1D );
      if (gridOffsetTable0!=USE_STEPS)
        tmp_gradCoefficients += bot;
    }
  } // end MultiplyJacobianWithValue()
}; // end class


/** \class RecursiveBSplineImplementation_MultiplyJacobianWithValue
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< class gradPointerType, unsigned int SplineOrder, class InputValueType >
class RecursiveBSplineImplementation_MultiplyJacobianWithValue< gradPointerType, 0, SplineOrder, InputValueType >
{
public:
  /** MultiplyJacobianWithValue recursive implementation. */
  static FORCEINLINE void MultiplyJacobianWithValue(
    gradPointerType gradCoefficients,
    double weight,
    const InputValueType gradValue,
    const OffsetValueType * gridOffsetTable,
    const double * weights1D )
  {
    *gradCoefficients += weight * gradValue;
  } // end MultiplyJacobianWithValue()
}; // end class

} // end namespace itk




/** \class RecursiveBSplineImplementation_GetJacobianOfSpatialJacobian
 *
 * \brief Define general case
 */

template< class OutputPointerType, unsigned int SpaceDimension, unsigned int jsj_length, unsigned int SplineOrder, class InputPointerType >
class RecursiveBSplineImplementation_GetJacobianOfSpatialJacobian
{
public:
  typedef typename std::iterator_traits< InputPointerType >::value_type OutputValueType; //\todo: is this the proper use of std::iterator_traits? Preferably we use 'using std::iterator_traits', to allow custom template specializations.
  typedef OutputValueType * RecursiveOutputPointerType;

  /** Helper constant variable. */
  itkStaticConstMacro( HelperConstVariable, unsigned int,
    ( SpaceDimension - 1 ) * ( SplineOrder + 1 ) );

  /** GetSpatialJacobian recursive implementation. */
  static inline void GetJacobianOfSpatialJacobian(
    OutputPointerType jsj_out,
    const InputPointerType jsj,
    const double * weights1D,
    const double * derivativeWeights1D )
  {
    /** Create a temporary jsj.*/
    OutputValueType  tmp_jsj[ jsj_length + 1];

    for( unsigned int k = 0; k <= SplineOrder; ++k )
    {
      // Multiply by the weights
      tmp_jsj[ 0 ] = jsj[ 0 ] * weights1D[ k + HelperConstVariable ];
      // Multiply by the derivative weights
      tmp_jsj[ 1 ] = jsj[ 0 ] * derivativeWeights1D[ k + HelperConstVariable ];

      for( unsigned int n = 1; n < jsj_length; ++n )
      {
        tmp_jsj[ n+1 ] += jsj[ n ] * weights1D[ k + HelperConstVariable ];
      };

      RecursiveBSplineImplementation_GetJacobianOfSpatialJacobian< OutputPointerType, SpaceDimension - 1, jsj_length +1 , SplineOrder, InputPointerType >
        ::GetJacobianOfSpatialJacobian( jsj_out, &tmp_jsj[0], weights1D , derivativeWeights1D );

    }
  } // end GetJacobianOfSpatialJacobian()
}; // end class


/** \class RecursiveBSplineImplementation_GetJacobianOfSpatialJacobian
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template< class OutputPointerType, unsigned int jsj_length, unsigned int SplineOrder, class InputPointerType >
class RecursiveBSplineImplementation_GetJacobianOfSpatialJacobian< OutputPointerType, 0, jsj_length, SplineOrder, InputPointerType >
{
public:
  /** GetSpatialJacobian recursive implementation. */
  static inline void GetJacobianOfSpatialJacobian(
    OutputPointerType & jsj_out,
    const InputPointerType jsj,
    const double * weights1D,
    const double * derivativeWeights1D )
  {
    InputPointerType jsj_iterator = jsj;
    for (int i = 0 ; i < jsj_length; ++i ){
      *jsj_out = *jsj_iterator;
      ++jsj_out; ++jsj_iterator;
    }
  } // end GetJacobianOfSpatialJacobian()


}; // end class


#undef FORCEINLINE // remove temporary preprocessor definition

#endif /* __itkRecursiveBSplineImplementation_h */
