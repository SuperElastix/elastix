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
#ifndef itkRecursiveBSplineTransformImplementation_h
#define itkRecursiveBSplineTransformImplementation_h

#include "itkRecursiveBSplineInterpolationWeightFunction.h"

// Standard C++ header files:
#include <algorithm> // For copy_n and fill_n.
#include <cassert>
#include <cstring> // For memcpy.

namespace itk
{

/** \class RecursiveBSplineTransformImplementation
 *
 * \brief This helper class contains the actual implementation of the
 * recursive B-spline transform
 *
 * Compared to the RecursiveBSplineTransformImplementation class, this
 * class works as a vector operator, and is therefore also templated
 * over the OutputDimension.
 *
 * Note: More optimized code can be found in itkRecursiveBSplineImplementation.h
 *
 * \ingroup ITKTransform
 */

template <unsigned int OutputDimension, unsigned int SpaceDimension, unsigned int SplineOrder, class TScalar>
class ITK_TEMPLATE_EXPORT RecursiveBSplineTransformImplementation
{
public:
  using InternalFloatType = double;

  /** Helper constant variable. */
  itkStaticConstMacro(HelperConstVariable, unsigned int, (SpaceDimension - 1) * (SplineOrder + 1));

  /** Typedef to know the number of indices at compile time. */
  using RecursiveBSplineWeightFunctionType =
    itk::RecursiveBSplineInterpolationWeightFunction<TScalar, OutputDimension, SplineOrder>;
  itkStaticConstMacro(BSplineNumberOfIndices, unsigned int, RecursiveBSplineWeightFunctionType::NumberOfIndices);

  /** TransformPoint recursive implementation. */
  static inline void
  TransformPoint(TScalar * const               opp,
                 const TScalar * const * const mu,
                 const OffsetValueType * const gridOffsetTable,
                 const double * const          weights1D)
  {
    /** Make a copy of the pointers to mu. The pointer will move later. */
    const TScalar * tmp_mu[OutputDimension];
    std::copy_n(mu, OutputDimension, tmp_mu);

    /** Create a temporary opp and initialize the original. */
    TScalar tmp_opp[OutputDimension];
    std::fill_n(opp, OutputDimension, 0.0);

    const OffsetValueType bot = gridOffsetTable[SpaceDimension - 1];
    for (unsigned int k = 0; k <= SplineOrder; ++k)
    {
      /** Recurse. */
      RecursiveBSplineTransformImplementation<OutputDimension, SpaceDimension - 1, SplineOrder, TScalar>::
        TransformPoint(tmp_opp, tmp_mu, gridOffsetTable, weights1D);

      /** Accumulate the weights. */
      for (unsigned int j = 0; j < OutputDimension; ++j)
      {
        opp[j] += tmp_opp[j] * weights1D[k + HelperConstVariable];

        // move to the next mu
        tmp_mu[j] += bot;
      }
    }
  } // end TransformPoint()


  /** GetJacobian recursive implementation. */
  static inline void
  GetJacobian(TScalar *& jacobians, const double * const weights1D, const double value)
  {
    for (unsigned int k = 0; k <= SplineOrder; ++k)
    {
      /** Recurse. */
      RecursiveBSplineTransformImplementation<OutputDimension, SpaceDimension - 1, SplineOrder, TScalar>::GetJacobian(
        jacobians, weights1D, value * weights1D[k + HelperConstVariable]);
    }
  } // end GetJacobian()


  /** EvaluateJacobianWithImageGradientProduct recursive implementation. */
  static inline void
  EvaluateJacobianWithImageGradientProduct(TScalar *&                      imageJacobian,
                                           const InternalFloatType * const movingImageGradient,
                                           const double * const            weights1D,
                                           const double                    value)
  {
    for (unsigned int k = 0; k <= SplineOrder; ++k)
    {
      /** Recurse. */
      RecursiveBSplineTransformImplementation<OutputDimension, SpaceDimension - 1, SplineOrder, TScalar>::
        EvaluateJacobianWithImageGradientProduct(
          imageJacobian, movingImageGradient, weights1D, value * weights1D[k + HelperConstVariable]);
    }
  } // end EvaluateJacobianWithImageGradientProduct()


  /** ComputeNonZeroJacobianIndices recursive implementation. */
  static inline void
  ComputeNonZeroJacobianIndices(unsigned long *&              nzji,
                                const unsigned long           parametersPerDim,
                                unsigned long                 currentIndex,
                                const OffsetValueType * const gridOffsetTable)
  {
    const OffsetValueType bot = gridOffsetTable[SpaceDimension - 1];
    for (unsigned int k = 0; k <= SplineOrder; ++k)
    {
      /** Recurse. */
      RecursiveBSplineTransformImplementation<OutputDimension, SpaceDimension - 1, SplineOrder, TScalar>::
        ComputeNonZeroJacobianIndices(nzji, parametersPerDim, currentIndex, gridOffsetTable);

      currentIndex += bot;
    }
  } // end ComputeNonZeroJacobianIndices()


  /** GetSpatialJacobian recursive implementation.
   * As an (almost) free by-product this function delivers the displacement,
   * i.e. the TransformPoint() function.
   */
  static inline void
  GetSpatialJacobian(InternalFloatType * const     sj,
                     const TScalar * const * const mu,
                     const OffsetValueType * const gridOffsetTable,
                     const double * const          weights1D,  // normal B-spline weights
                     const double * const          derivativeWeights1D) // 1st derivative of B-spline
  {
    /** Make a copy of the pointers to mu. The pointer will move later. */
    const TScalar * tmp_mu[OutputDimension];
    std::copy_n(mu, OutputDimension, tmp_mu);

    /** Create a temporary sj and initialize the original. */
    InternalFloatType tmp_sj[OutputDimension * SpaceDimension];
    for (unsigned int n = 0; n < OutputDimension * (SpaceDimension + 1); ++n)
    {
      sj[n] = 0.0;
    }

    const OffsetValueType bot = gridOffsetTable[SpaceDimension - 1];
    for (unsigned int k = 0; k <= SplineOrder; ++k)
    {
      /** Recurse. */
      RecursiveBSplineTransformImplementation<OutputDimension, SpaceDimension - 1, SplineOrder, TScalar>::
        GetSpatialJacobian(tmp_sj, tmp_mu, gridOffsetTable, weights1D, derivativeWeights1D);

      /** Accumulate the weights part. */
      for (unsigned int n = 0; n < OutputDimension * SpaceDimension; ++n)
      {
        sj[n] += tmp_sj[n] * weights1D[k + HelperConstVariable];
      }

      /** Accumulate the derivative weights part. */
      for (unsigned int j = 0; j < OutputDimension; ++j)
      {
        sj[OutputDimension * SpaceDimension + j] += tmp_sj[j] * derivativeWeights1D[k + HelperConstVariable];

        // move to the next mu
        tmp_mu[j] += bot;
      }
    }
  } // end GetSpatialJacobian()


  /** GetSpatialHessian recursive implementation.
   * As an (almost) free by-product this function delivers the displacement,
   * i.e. the TransformPoint() function, as well as the SpatialJacobian.
   *
   * Specifically, sh is the output argument. It should be allocated with a size
   * OutputDimension * ( SpaceDimension + 1 ) * ( SpaceDimension + 2 ) / 2.
   * sh should point to allocated memory, but this function initializes sh.
   *
   * Upon return sh contains the spatial Hessian, spatial Jacobian and transformpoint. With
   * Hk = [ transformPoint     spatialJacobian'
   *        spatialJacobian    spatialHessian   ] .
   * (Hk specifies all info of dimension (element) k (< OutputDimension) of the point
   * and spatialJacobian is a vector of the derivative of this point with respect to the dimensions.)
   * The i,j (both < SpaceDimension) element of Hk is stored in:
   * i<=j : sh[ k +  OutputDimension * (i + j*(j+1)/2 ) ]
   * i>=j : sh[ k +  OutputDimension * (j + i*(i+1)/2 ) ]
   *
   * Note that we store only one of the symmetric halves of Hk.
   */
  static inline void
  GetSpatialHessian(InternalFloatType * const     sh,
                    const TScalar * const * const mu,
                    const OffsetValueType * const gridOffsetTable,
                    const double * const          weights1D,           // normal B-spline weights
                    const double * const          derivativeWeights1D, // 1st derivative of B-spline
                    const double * const          hessianWeights1D)             // 2nd derivative of B-spline
  {
    const unsigned int helperDim1 = OutputDimension * SpaceDimension * (SpaceDimension + 1) / 2;
    const unsigned int helperDim2 = OutputDimension * (SpaceDimension + 1) * (SpaceDimension + 2) / 2;

    /** Make a copy of the pointers to mu. The pointer will move later. */
    const TScalar * tmp_mu[OutputDimension];
    std::copy_n(mu, OutputDimension, tmp_mu);

    /** Create a temporary sh and initialize the original. */
    InternalFloatType tmp_sh[helperDim1];
    for (unsigned int n = 0; n < helperDim2; ++n)
    {
      sh[n] = 0.0;
    }

    const OffsetValueType bot = gridOffsetTable[SpaceDimension - 1];
    for (unsigned int k = 0; k <= SplineOrder; ++k)
    {
      /** Recurse. */
      RecursiveBSplineTransformImplementation<OutputDimension, SpaceDimension - 1, SplineOrder, TScalar>::
        GetSpatialHessian(tmp_sh, tmp_mu, gridOffsetTable, weights1D, derivativeWeights1D, hessianWeights1D);

      /** Accumulate the weights part. */
      for (unsigned int n = 0; n < helperDim1; ++n)
      {
        sh[n] += tmp_sh[n] * weights1D[k + HelperConstVariable];
      }

      /** Accumulate the derivative weights part. */
      for (unsigned int n = 0; n < SpaceDimension; ++n)
      {
        for (unsigned int j = 0; j < OutputDimension; ++j)
        {
          sh[OutputDimension * n + helperDim1 + j] +=
            tmp_sh[OutputDimension * n * (n + 1) / 2 + j] * derivativeWeights1D[k + HelperConstVariable];
        }
      }

      /** Accumulate the Hessian weights part. */
      for (unsigned int j = 0; j < OutputDimension; ++j)
      {
        sh[helperDim2 - OutputDimension + j] += tmp_sh[j] * hessianWeights1D[k + HelperConstVariable];

        // move to the next mu
        tmp_mu[j] += bot;
      }
    }
  } // end GetSpatialHessian()


  /** GetJacobianOfSpatialJacobian recursive implementation.
   * Multiplication with the direction cosines is performed in the end-case.
   */
  static inline void
  GetJacobianOfSpatialJacobian(InternalFloatType *&            jsj_out,
                               const double * const            weights1D,           // normal B-spline weights
                               const double * const            derivativeWeights1D, // 1st derivative of B-spline
                               const double * const            directionCosines,
                               const InternalFloatType * const jsj)
  {
    const unsigned int helperDim = OutputDimension - SpaceDimension + 1;

    /** Create a temporary jsj. Here, an additional element is needed for the Jacobian. */
    InternalFloatType tmp_jsj[helperDim + 1];

    for (unsigned int k = 0; k <= SplineOrder; ++k)
    {
      const double w = weights1D[k + HelperConstVariable];
      const double dw = derivativeWeights1D[k + HelperConstVariable];

      /** Initialize the weights part of the temporary jsj. */
      for (unsigned int n = 0; n < helperDim; ++n)
      {
        tmp_jsj[n] = jsj[n] * w;
      }

      /** Initialize the derivative weights part. */
      tmp_jsj[helperDim] = jsj[0] * dw;

      /** Recurse. */
      RecursiveBSplineTransformImplementation<OutputDimension, SpaceDimension - 1, SplineOrder, TScalar>::
        GetJacobianOfSpatialJacobian(jsj_out, weights1D, derivativeWeights1D, directionCosines, tmp_jsj);
    }
  } // end GetJacobianOfSpatialJacobian()


  /** GetJacobianOfSpatialHessian recursive implementation.
   * Multiplication with the direction cosines is performed in the end - case.
   */
  static inline void
  GetJacobianOfSpatialHessian(InternalFloatType *&            jsh_out,
                              const double * const            weights1D,           // normal B-spline weights
                              const double * const            derivativeWeights1D, // 1st derivative of B-spline
                              const double * const            hessianWeights1D,    // 2nd derivative of B-spline
                              const double * const            directionCosines,
                              const InternalFloatType * const jsh)
  {
    const unsigned int helperDim = OutputDimension - SpaceDimension;
    const unsigned int helperDimW = (helperDim + 1) * (helperDim + 2) / 2;
    const unsigned int helperDimDW = helperDim + 1;

    /** Create a temporary jsh. */
    InternalFloatType tmp_jsh[helperDimW + helperDimDW + 1];

    for (unsigned int k = 0; k <= SplineOrder; ++k)
    {
      /** Store some weights. */
      const double w = weights1D[k + HelperConstVariable];
      const double dw = derivativeWeights1D[k + HelperConstVariable];
      const double hw = hessianWeights1D[k + HelperConstVariable];

      /** Initialize the weights part of the temporary jsh. */
      for (unsigned int n = 0; n < helperDimW; ++n)
      {
        tmp_jsh[n] = jsh[n] * w;
      }

      /** Initialize the derivative weights part. */
      for (unsigned int n = 0; n < helperDimDW; ++n)
      {
        unsigned int nn = n * (n + 1) / 2;
        tmp_jsh[n + helperDimW] = jsh[nn] * dw;
      }

      /** Initialize the Hessian weights part. */
      tmp_jsh[helperDimW + helperDimDW] = jsh[0] * hw;

      /** Recurse. */
      RecursiveBSplineTransformImplementation<OutputDimension, SpaceDimension - 1, SplineOrder, TScalar>::
        GetJacobianOfSpatialHessian(
          jsh_out, weights1D, derivativeWeights1D, hessianWeights1D, directionCosines, tmp_jsh);
    }
  } // end GetJacobianOfSpatialHessian()
};


/** \class RecursiveBSplineTransformImplementation
 *
 * \brief Define the end case for SpaceDimension = 0.
 */

template <unsigned int OutputDimension, unsigned int SplineOrder, class TScalar>
class ITK_TEMPLATE_EXPORT RecursiveBSplineTransformImplementation<OutputDimension, 0, SplineOrder, TScalar>
{
public:
  using InternalFloatType = double;

  /** Typedef to know the number of indices at compile time. */
  using RecursiveBSplineWeightFunctionType =
    itk::RecursiveBSplineInterpolationWeightFunction<TScalar, OutputDimension, SplineOrder>;
  itkStaticConstMacro(BSplineNumberOfIndices, unsigned int, RecursiveBSplineWeightFunctionType::NumberOfIndices);

  /** TransformPoint recursive implementation. */
  static inline void
  TransformPoint(TScalar * const               opp,
                 const TScalar * const * const mu,
                 const OffsetValueType * const gridOffsetTable,
                 const double * const          weights1D)
  {
    for (unsigned int j = 0; j < OutputDimension; ++j)
    {
      opp[j] = *(mu[j]);
    }
  } // end TransformPoint()


  /** GetJacobian recursive implementation. */
  static inline void
  GetJacobian(TScalar *& jacobians, const double * const weights1D, const double value)
  {
    unsigned long offset = 0;
    for (unsigned int j = 0; j < OutputDimension; ++j)
    {
      offset = j * BSplineNumberOfIndices * (OutputDimension + 1);
      jacobians[offset] = value;
    }
    ++jacobians;
  } // end GetJacobian()


  /** EvaluateJacobianWithImageGradientProduct recursive implementation. */
  static inline void
  EvaluateJacobianWithImageGradientProduct(TScalar *&                      imageJacobian,
                                           const InternalFloatType * const movingImageGradient,
                                           const double * const            weights1D,
                                           const double                    value)
  {
    for (unsigned int j = 0; j < OutputDimension; ++j)
    {
      *(imageJacobian + j * BSplineNumberOfIndices) = value * movingImageGradient[j];
    }
    ++imageJacobian;
  } // end EvaluateJacobianWithImageGradientProduct()


  /** ComputeNonZeroJacobianIndices recursive implementation. */
  static inline void
  ComputeNonZeroJacobianIndices(unsigned long *&              nzji,
                                const unsigned long           parametersPerDim,
                                const unsigned long           currentIndex,
                                const OffsetValueType * const gridOffsetTable)
  {
    for (unsigned int j = 0; j < OutputDimension; ++j)
    {
      nzji[j * BSplineNumberOfIndices] = currentIndex + j * parametersPerDim;
    }
    ++nzji;
  } // end ComputeNonZeroJacobianIndices()


  /** GetSpatialJacobian recursive implementation. */
  static inline void
  GetSpatialJacobian(InternalFloatType * const     sj,
                     const TScalar * const * const mu,
                     const OffsetValueType * const gridOffsetTable,
                     const double * const          weights1D,  // normal B-spline weights
                     const double * const          derivativeWeights1D) // 1st derivative of B-spline
  {
    for (unsigned int j = 0; j < OutputDimension; ++j)
    {
      sj[j] = *(mu[j]);
    }
  } // end GetSpatialJacobian()


  /** GetSpatialHessian recursive implementation. */
  static inline void
  GetSpatialHessian(InternalFloatType * const     sh,
                    const TScalar * const * const mu,
                    const OffsetValueType * const gridOffsetTable,
                    const double * const          weights1D,           // normal B-spline weights
                    const double * const          derivativeWeights1D, // 1st derivative of B-spline
                    const double * const          hessianWeights1D)             // 2nd derivative of B-spline
  {
    for (unsigned int j = 0; j < OutputDimension; ++j)
    {
      sh[j] = *(mu[j]);
    }
  } // end GetSpatialHessian()


  /** GetJacobianOfSpatialJacobian recursive implementation. */
  static inline void
  GetJacobianOfSpatialJacobian(InternalFloatType *&            jsj_out,
                               const double * const            weights1D,           // normal B-spline weights
                               const double * const            derivativeWeights1D, // 1st derivative of B-spline
                               const double * const            directionCosines,
                               const InternalFloatType * const jsj)
  {
    /** Copy the correct elements to the output.
     * Note that the first element jsj[0] is the normal Jacobian. We ignore it for now.
     * Also note that the received order is [dz, dy, dx] and that we return [dx, dy, dz].
     * Returns full jsj
     */
    for (unsigned int j = 0; j < OutputDimension; ++j)
    {
      jsj_out[j] = jsj[OutputDimension] * directionCosines[j];
      for (unsigned int k = 1; k < OutputDimension; ++k)
      {
        jsj_out[k] += jsj[OutputDimension - k] * directionCosines[k * OutputDimension + j];
      }
    }

    /** Mirror the results. */
    unsigned int offset = 0;
    for (unsigned int i = 0; i < OutputDimension; ++i)
    {
      offset = i * (OutputDimension * (BSplineNumberOfIndices * OutputDimension + 1));
      for (unsigned int j = 0; j < OutputDimension; ++j)
      {
        jsj_out[j + offset] = jsj_out[j];
      }
    }

    /** Jump to the next non-empty matrix, skipping the zero matrices. */
    jsj_out += OutputDimension * OutputDimension;

  } // end GetJacobianOfSpatialJacobian()


  /** GetJacobianOfSpatialHessian recursive implementation. */
  static inline void
  GetJacobianOfSpatialHessian(InternalFloatType *&            jsh_out,
                              const double * const            weights1D,           // normal B-spline weights
                              const double * const            derivativeWeights1D, // 1st derivative of B-spline
                              const double * const            hessianWeights1D,    // 2nd derivative of B-spline
                              const double * const            directionCosines,
                              const InternalFloatType * const jsh)
  {
    double jsh_tmp[OutputDimension * OutputDimension];
    double matrixProduct[OutputDimension * OutputDimension];

    /** Copy the correct elements to the intermediate matrix.
     * Note that in contrast to the other function, here we create the full matrix.
     *
     * For dimensions 2 and 3 optimized code (loop unrolling) is provided. Smart compilers may
     * not need that.
     */
    if (OutputDimension == 3)
    {
      const double tmp[] = { jsh[9], jsh[8], jsh[7], jsh[8], jsh[5], jsh[4], jsh[7], jsh[4], jsh[2] };
      FastBitwiseCopy(jsh_tmp, tmp);
    }
    else if (OutputDimension == 2)
    {
      const double tmp[] = { jsh[5], jsh[4], jsh[4], jsh[2] };
      FastBitwiseCopy(jsh_tmp, tmp);
    }
    else // the general case
    {
      for (unsigned int j = 0; j < OutputDimension; ++j)
      {
        for (unsigned int i = 0; i <= j; ++i)
        {
          jsh_tmp[j * OutputDimension + i] =
            jsh[(OutputDimension - j) + (OutputDimension - i) * (OutputDimension - i + 1) / 2];
          if (i != j)
          {
            jsh_tmp[i * OutputDimension + j] = jsh_tmp[j * OutputDimension + i];
          }
        }
      }
    }

    /** Pre-multiply directionCosines^t * H. */
    for (unsigned int i = 0; i < OutputDimension; ++i) // row
    {
      for (unsigned int j = 0; j < OutputDimension; ++j) // column
      {
        double accum = directionCosines[i] * jsh_tmp[j];
        for (unsigned int k = 1; k < OutputDimension; ++k)
        {
          accum += directionCosines[k * OutputDimension + i] * jsh_tmp[k * OutputDimension + j];
        }
        matrixProduct[i * OutputDimension + j] = accum;
      }
    }

    /** Post-multiply matrixProduct * directionCosines. */
    for (unsigned int i = 0; i < OutputDimension; ++i) // row
    {
      for (unsigned int j = 0; j < OutputDimension; ++j) // column
      {
        double accum = matrixProduct[i * OutputDimension] * directionCosines[j];
        for (unsigned int k = 1; k < OutputDimension; ++k)
        {
          accum += matrixProduct[i * OutputDimension + k] * directionCosines[k * OutputDimension + j];
        }
        jsh_out[i * OutputDimension + j] = accum;
      }
    }

    /** Mirror the results. */
    unsigned long offset = 0;
    for (unsigned int i = 0; i < OutputDimension; ++i)
    {
      offset = i * (OutputDimension * OutputDimension * (BSplineNumberOfIndices * OutputDimension + 1));
      for (unsigned int j = 0; j < OutputDimension * OutputDimension; ++j)
      {
        jsh_out[j + offset] = jsh_out[j];
      }
    }

    /** Jump to the next non-empty matrix, skipping the zero matrices. */
    jsh_out += OutputDimension * OutputDimension * OutputDimension;

  } // end GetJacobianOfSpatialHessian()


private:
  template <typename T>
  static void
  FastBitwiseCopy(T & destination, const T & source)
  {
    std::memcpy(&destination, &source, sizeof(T));
  }

  template <typename T1, typename T2>
  static void
  FastBitwiseCopy(const T1 &, const T2 &)
  {
    assert(!"This FastBitwiseCopy overload should not be called!");
  }
};


} // end namespace itk

#endif /* itkRecursiveBSplineTransformImplementation_h */
