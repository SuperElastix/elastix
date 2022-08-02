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
#ifndef itkTransformBendingEnergyPenaltyTerm_hxx
#define itkTransformBendingEnergyPenaltyTerm_hxx

#include "itkTransformBendingEnergyPenaltyTerm.h"

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template <class TFixedImage, class TScalarType>
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::TransformBendingEnergyPenaltyTerm()
{
  /** Initialize member variables. */

  /** Turn on the sampler functionality. */
  this->SetUseImageSampler(true);

  this->m_NumberOfSamplesForSelfHessian = 100000;

} // end Constructor


/**
 * ****************** GetValue *******************************
 */

template <class TFixedImage, class TScalarType>
auto
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetValue(const ParametersType & parameters) const
  -> MeasureType
{
  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  RealType           measure = NumericTraits<RealType>::Zero;
  SpatialHessianType spatialHessian;

  /** Check if the SpatialHessian is nonzero. */
  if (!this->m_AdvancedTransform->GetHasNonZeroSpatialHessian())
  {
    return static_cast<MeasureType>(measure);
  }

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image samples to calculate the penalty term. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    if (sampleOk)
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the spatial Hessian of the transformation at the current point.
       * This is needed to compute the bending energy.
       */
      this->m_AdvancedTransform->GetSpatialHessian(fixedPoint, spatialHessian);

      /** Compute the contribution of this point. */
      for (unsigned int k = 0; k < FixedImageDimension; ++k)
      {
        measure += vnl_math::sqr(spatialHessian[k].GetVnlMatrix().frobenius_norm());
      }

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Update measure value. */
  measure /= static_cast<RealType>(this->m_NumberOfPixelsCounted);

  /** Return the value. */
  return static_cast<MeasureType>(measure);

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedImage, class TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetDerivative(const ParametersType & parameters,
                                                                           DerivativeType &       derivative) const
{
  /** Slower, but works. */
  MeasureType dummyvalue = NumericTraits<MeasureType>::Zero;
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ****************** GetValueAndDerivativeSingleThreaded *******************************
 */

template <class TFixedImage, class TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetValueAndDerivativeSingleThreaded(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  /** Create and initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  RealType measure = NumericTraits<RealType>::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits<DerivativeValueType>::ZeroValue());

  SpatialHessianType           spatialHessian;
  JacobianOfSpatialHessianType jacobianOfSpatialHessian;
  NonZeroJacobianIndicesType   nonZeroJacobianIndices;
  const NumberOfParametersType numberOfNonZeroJacobianIndices =
    this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  jacobianOfSpatialHessian.resize(numberOfNonZeroJacobianIndices);
  nonZeroJacobianIndices.resize(numberOfNonZeroJacobianIndices);

  /** Check if the SpatialHessian is nonzero. */
  if (!this->m_AdvancedTransform->GetHasNonZeroSpatialHessian() &&
      !this->m_AdvancedTransform->GetHasNonZeroJacobianOfSpatialHessian())
  {
    value = static_cast<MeasureType>(measure);
    return;
  }
  // TODO: This is only required once! and not every iteration.

  /** Check if this transform is a B-spline transform. */
  typename BSplineOrder3TransformType::Pointer dummy; // default-constructed (null)
  bool                                         transformIsBSpline = this->CheckForBSplineTransform2(dummy);

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the penalty term and its derivative. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Although the mapped point is not needed to compute the penalty term,
     * we compute in order to check if it maps inside the support region of
     * the B-spline and if it maps inside the moving image mask.
     */

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    if (sampleOk)
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the spatial Hessian of the transformation at the current point.
       * This is needed to compute the bending energy.
       */
      //       this->m_AdvancedTransform->GetSpatialHessian( fixedPoint,
      //         spatialHessian );
      //       this->m_AdvancedTransform->GetJacobianOfSpatialHessian( fixedPoint,
      //         jacobianOfSpatialHessian, nonZeroJacobianIndices );
      this->m_AdvancedTransform->GetJacobianOfSpatialHessian(
        fixedPoint, spatialHessian, jacobianOfSpatialHessian, nonZeroJacobianIndices);

      /** Prepare some stuff for the computation of the metric (derivative). */
      FixedArray<InternalMatrixType, FixedImageDimension> A;
      for (unsigned int k = 0; k < FixedImageDimension; ++k)
      {
        A[k] = spatialHessian[k].GetVnlMatrix();
      }

      /** Compute the contribution to the metric value of this point. */
      for (unsigned int k = 0; k < FixedImageDimension; ++k)
      {
        measure += vnl_math::sqr(A[k].frobenius_norm());
      }

      /** Make a distinction between a B-spline transform and other transforms. */
      if (!transformIsBSpline)
      {
        /** Compute the contribution to the metric derivative of this point. */
        for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
        {
          for (unsigned int k = 0; k < FixedImageDimension; ++k)
          {
            /** This computes:
             * \sum_i \sum_j A_ij B_ij = element_product(A,B).mean()*B.size()
             */
            const InternalMatrixType & B = jacobianOfSpatialHessian[mu][k].GetVnlMatrix();

            RealType                                    matrixProduct = 0.0;
            typename InternalMatrixType::const_iterator itA = A[k].begin();
            typename InternalMatrixType::const_iterator itB = B.begin();
            typename InternalMatrixType::const_iterator itAend = A[k].end();
            while (itA != itAend)
            {
              matrixProduct += (*itA) * (*itB);
              ++itA;
              ++itB;
            }

            derivative[nonZeroJacobianIndices[mu]] += 2.0 * matrixProduct;
          }
        }
      }
      else
      {
        /** For the B-spline transform we know that only 1/FixedImageDimension
         * part of the JacobianOfSpatialHessian is non-zero.
         */

        /** Compute the contribution to the metric derivative of this point. */
        unsigned int numParPerDim = nonZeroJacobianIndices.size() / FixedImageDimension;
        /*SpatialHessianType * basepointer1 = &jacobianOfSpatialHessian[ 0 ];
        unsigned long * basepointer2 = &nonZeroJacobianIndices[ 0 ];
        double * basepointer3 = &derivative[ 0 ];*/
        for (unsigned int mu = 0; mu < numParPerDim; ++mu)
        {
          for (unsigned int k = 0; k < FixedImageDimension; ++k)
          {
            /** This computes:
             * \sum_i \sum_j A_ij B_ij = element_product(A,B).mean()*B.size()
             */
            /*const InternalMatrixType & B
              = (*( basepointer1 + mu + numParPerDim * k ))[ k ].GetVnlMatrix();
            const RealType matrixMean = element_product( A[ k ], B ).mean();
            *( basepointer3 + (*( basepointer2 + mu + numParPerDim * k )) )
              += 2.0 * matrixMean * Bsize;*/
            const InternalMatrixType & B = jacobianOfSpatialHessian[mu + numParPerDim * k][k].GetVnlMatrix();

            RealType                                    matrixElementProduct = 0.0;
            typename InternalMatrixType::const_iterator itA = A[k].begin();
            typename InternalMatrixType::const_iterator itB = B.begin();
            typename InternalMatrixType::const_iterator itAend = A[k].end();
            while (itA != itAend)
            {
              matrixElementProduct += (*itA) * (*itB);
              ++itA;
              ++itB;
            }

            derivative[nonZeroJacobianIndices[mu + numParPerDim * k]] += 2.0 * matrixElementProduct;
          }
        }
      } // end if B-spline

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Update measure value. */
  measure /= static_cast<RealType>(this->m_NumberOfPixelsCounted);
  derivative /= static_cast<RealType>(this->m_NumberOfPixelsCounted);

  /** The return value. */
  value = static_cast<MeasureType>(measure);

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetValueAndDerivative(const ParametersType & parameters,
                                                                                   MeasureType &          value,
                                                                                   DerivativeType & derivative) const
{
  /** Option for now to still use the single threaded code. */
  if (!this->m_UseMultiThread)
  {
    return this->GetValueAndDerivativeSingleThreaded(parameters, value, derivative);
  }

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Launch multi-threading metric */
  this->LaunchGetValueAndDerivativeThreaderCallback();

  /** Gather the metric values and derivatives from all threads. */
  this->AfterThreadedGetValueAndDerivative(value, derivative);

} // end GetValueAndDerivative()


/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::ThreadedGetValueAndDerivative(ThreadIdType threadId)
{
  /** Create and initialize some variables. */
  SpatialHessianType           spatialHessian;
  JacobianOfSpatialHessianType jacobianOfSpatialHessian;
  NonZeroJacobianIndicesType   nonZeroJacobianIndices;
  const NumberOfParametersType numberOfNonZeroJacobianIndices =
    this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  jacobianOfSpatialHessian.resize(numberOfNonZeroJacobianIndices);
  nonZeroJacobianIndices.resize(numberOfNonZeroJacobianIndices);

  /** Check if the SpatialHessian is nonzero. */
  if (!this->m_AdvancedTransform->GetHasNonZeroSpatialHessian() &&
      !this->m_AdvancedTransform->GetHasNonZeroJacobianOfSpatialHessian())
  {
    return;
  }
  // TODO: This is only required once! and not every iteration.

  /** Check if this transform is a B-spline transform. */
  typename BSplineOrder3TransformType::Pointer dummy; // default-constructed (null)
  bool                                         transformIsBSpline = this->CheckForBSplineTransform2(dummy);

  /** Get a handle to the pre-allocated derivative for the current thread.
   * The initialization is performed at the beginning of each resolution in
   * InitializeThreadingParameters(), and at the end of each iteration in
   * AfterThreadedGetValueAndDerivative() and the accumulate functions.
   */
  DerivativeType & derivative = this->m_GetValueAndDerivativePerThreadVariables[threadId].st_Derivative;

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end = nrOfSamplesPerThreads * (threadId + 1);
  pos_begin = (pos_begin > sampleContainerSize) ? sampleContainerSize : pos_begin;
  pos_end = (pos_end > sampleContainerSize) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->Begin();
  fbegin += (int)pos_begin;
  fend += (int)pos_end;

  /** Create variables to store intermediate results. circumvent false sharing */
  unsigned long numberOfPixelsCounted = 0;
  MeasureType   measure = NumericTraits<MeasureType>::Zero;

  /** Loop over the fixed image to calculate the penalty term and its derivative. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Although the mapped point is not needed to compute the penalty term,
     * we compute in order to check if it maps inside the support region of
     * the B-spline and if it maps inside the moving image mask.
     */

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    if (sampleOk)
    {
      ++numberOfPixelsCounted;

      /** Get the spatial Hessian of the transformation at the current point.
       * This is needed to compute the bending energy.
       */
      this->m_AdvancedTransform->GetJacobianOfSpatialHessian(
        fixedPoint, spatialHessian, jacobianOfSpatialHessian, nonZeroJacobianIndices);

      /** Prepare some stuff for the computation of the metric (derivative). */
      FixedArray<InternalMatrixType, FixedImageDimension> A;
      for (unsigned int k = 0; k < FixedImageDimension; ++k)
      {
        A[k] = spatialHessian[k].GetVnlMatrix();
      }

      /** Compute the contribution to the metric value of this point. */
      for (unsigned int k = 0; k < FixedImageDimension; ++k)
      {
        measure += vnl_math::sqr(A[k].frobenius_norm());
      }

      /** Make a distinction between a B-spline transform and other transforms. */
      if (!transformIsBSpline)
      {
        /** Compute the contribution to the metric derivative of this point. */
        for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
        {
          for (unsigned int k = 0; k < FixedImageDimension; ++k)
          {
            /** This computes:
             * \sum_i \sum_j A_ij B_ij = element_product(A,B).mean()*B.size()
             */
            const InternalMatrixType & B = jacobianOfSpatialHessian[mu][k].GetVnlMatrix();

            RealType                                    matrixElementProduct = 0.0;
            typename InternalMatrixType::const_iterator itA = A[k].begin();
            typename InternalMatrixType::const_iterator itB = B.begin();
            typename InternalMatrixType::const_iterator itAend = A[k].end();
            while (itA != itAend)
            {
              matrixElementProduct += (*itA) * (*itB);
              ++itA;
              ++itB;
            }

            derivative[nonZeroJacobianIndices[mu]] += 2.0 * matrixElementProduct;
          }
        }
      }
      else
      {
        /** For the B-spline transform we know that only 1/FixedImageDimension
         * part of the JacobianOfSpatialHessian is non-zero.
         *
         * In addition we know that jsh[ mu + numParPerDim * k ][ k ] is the same for all k.
         */

        /** Compute the contribution to the metric derivative of this point. */
        const unsigned int numParPerDim = nonZeroJacobianIndices.size() / FixedImageDimension;
        for (unsigned int mu = 0; mu < numParPerDim; ++mu)
        {
          const InternalMatrixType & B = jacobianOfSpatialHessian[mu + numParPerDim * 0][0].GetVnlMatrix();

          for (unsigned int k = 0; k < FixedImageDimension; ++k)
          {
            /** This computes:
             * \sum_i \sum_j A_ij B_ij = element_product(A,B).mean()*B.size()
             */
            RealType                                    matrixElementProduct = 0.0;
            typename InternalMatrixType::const_iterator itA = A[k].begin();
            typename InternalMatrixType::const_iterator itB = B.begin();
            typename InternalMatrixType::const_iterator itAend = A[k].end();
            while (itA != itAend)
            {
              matrixElementProduct += (*itA) * (*itB);
              ++itA;
              ++itB;
            }

            derivative[nonZeroJacobianIndices[mu + numParPerDim * k]] += 2.0 * matrixElementProduct;
          }
        }
      } // end if B-spline
    }   // end if sampleOk
  }     // end for loop over the image sample container

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  this->m_GetValueAndDerivativePerThreadVariables[threadId].st_NumberOfPixelsCounted = numberOfPixelsCounted;
  this->m_GetValueAndDerivativePerThreadVariables[threadId].st_Value = measure;

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::AfterThreadedGetValueAndDerivative(
  MeasureType &    value,
  DerivativeType & derivative) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted = 0;
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    this->m_NumberOfPixelsCounted += this->m_GetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted;

    /** Reset this variable for the next iteration. */
    this->m_GetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted = 0;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Accumulate and normalize values. */
  value = NumericTraits<MeasureType>::Zero;
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    value += this->m_GetValueAndDerivativePerThreadVariables[i].st_Value;

    /** Reset this variable for the next iteration. */
    this->m_GetValueAndDerivativePerThreadVariables[i].st_Value = NumericTraits<MeasureType>::Zero;
  }
  value /= static_cast<RealType>(this->m_NumberOfPixelsCounted);

  /** Accumulate derivatives. */
  // it seems that multi-threaded adding is faster than single-threaded
  // it seems that openmp is faster than itk threads
  // compute single-threadedly
  if (!this->m_UseMultiThread)
  {
    derivative = this->m_GetValueAndDerivativePerThreadVariables[0].st_Derivative;
    for (ThreadIdType i = 1; i < numberOfThreads; ++i)
    {
      derivative += this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative;
    }
    derivative /= static_cast<DerivativeValueType>(this->m_NumberOfPixelsCounted);
  }
  // compute multi-threadedly with itk threads
  else if (!this->m_UseOpenMP || true) // force
  {
    this->m_ThreaderMetricParameters.st_DerivativePointer = derivative.begin();
    this->m_ThreaderMetricParameters.st_NormalizationFactor =
      static_cast<DerivativeValueType>(this->m_NumberOfPixelsCounted);

    this->m_Threader->SetSingleMethod(this->AccumulateDerivativesThreaderCallback,
                                      const_cast<void *>(static_cast<const void *>(&this->m_ThreaderMetricParameters)));
    this->m_Threader->SingleMethodExecute();
  }
#ifdef ELASTIX_USE_OPENMP
  // compute multi-threadedly with openmp
  else
  {
    const DerivativeValueType numPix = static_cast<DerivativeValueType>(this->m_NumberOfPixelsCounted);
    const int                 nthreads = static_cast<int>(numberOfThreads);
    omp_set_num_threads(nthreads);
    const int spaceDimension = static_cast<int>(this->GetNumberOfParameters());
#  pragma omp parallel for
    for (int j = 0; j < spaceDimension; ++j)
    {
      DerivativeValueType tmp = NumericTraits<DerivativeValueType>::Zero;
      for (ThreadIdType i = 0; i < numberOfThreads; ++i)
      {
        tmp += this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative[j];
      }
      derivative[j] = tmp / numPix;
    }
  }
#endif

} // end AfterThreadedGetValueAndDerivative()


/**
 * ******************* GetSelfHessian *******************
 */

template <class TFixedImage, class TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetSelfHessian(const TransformParametersType & parameters,
                                                                            HessianType &                   H) const
{
  itkDebugMacro("GetSelfHessian()");

  using RowType = typename HessianType::row;
  using RowIteratorType = typename RowType::iterator;
  using ElementType = typename HessianType::pair_t;

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType   nonZeroJacobianIndices(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  JacobianOfSpatialHessianType jacobianOfSpatialHessian;

  /** Make sure the transform parameters are up to date. */
  // this->SetTransformParameters( parameters );

  const auto numberOfParameters = this->GetNumberOfParameters();

  /** Prepare Hessian */
  H.set_size(numberOfParameters, numberOfParameters);
  // H.Fill(0.0); //done by set_size for sparse matrix
  if (!this->m_AdvancedTransform->GetHasNonZeroJacobianOfSpatialHessian())
  {
    // H.fill_diagonal(1.0);
    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      H(i, i) = 1.0;
    }
    return;
  }

  /** Set up grid sampler */
  auto sampler = SelfHessianSamplerType::New();
  sampler->SetInputImageRegion(this->GetImageSampler()->GetInputImageRegion());
  sampler->SetMask(this->GetImageSampler()->GetMask());
  sampler->SetInput(this->GetFixedImage());
  sampler->SetNumberOfSamples(this->m_NumberOfSamplesForSelfHessian);

  /** Update the imageSampler and get a handle to the sample container. */
  sampler->Update();
  ImageSampleContainerPointer sampleContainer = sampler->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the d/dmu dT/dxdx terms. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Although the mapped point is not needed to compute the penalty term,
     * we compute in order to check if it maps inside the support region of
     * the B-spline and if it maps inside the moving image mask.
     */

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    if (sampleOk)
    {
      this->m_NumberOfPixelsCounted++;

      this->m_AdvancedTransform->GetJacobianOfSpatialHessian(
        fixedPoint, jacobianOfSpatialHessian, nonZeroJacobianIndices);

      /** Compute the contribution to the metric derivative of this point. */
      for (unsigned int muA = 0; muA < nonZeroJacobianIndices.size(); ++muA)
      {
        const unsigned int nmA = nonZeroJacobianIndices[muA];
        RowType &          rowVector = H.get_row(nmA);
        RowIteratorType    rowIt = rowVector.begin();

        for (unsigned int muB = muA; muB < nonZeroJacobianIndices.size(); ++muB)
        {
          const unsigned int nmB = nonZeroJacobianIndices[muB];

          RealType matrixProduct = 0.0;
          for (unsigned int k = 0; k < FixedImageDimension; ++k)
          {
            /** This computes:
             * \sum_i \sum_j A_ij B_ij = element_product(A,B).mean()*B.size()
             */
            const InternalMatrixType & A = jacobianOfSpatialHessian[muA][k].GetVnlMatrix();
            const InternalMatrixType & B = jacobianOfSpatialHessian[muB][k].GetVnlMatrix();

            typename InternalMatrixType::const_iterator itA = A.begin();
            typename InternalMatrixType::const_iterator itB = B.begin();
            typename InternalMatrixType::const_iterator itAend = A.end();
            while (itA != itAend)
            {
              matrixProduct += (*itA) * (*itB);
              ++itA;
              ++itB;
            }
          }

          /** Store at the right location in the H matrix.
           * Only upper triangular part is stored */

          /** Update hessian element */
          if ((matrixProduct > 1e-12) || (matrixProduct < 1e-12))
          {
            /**
             * H( nmA, nmB ) += 2.0 * matrixProduct;
             * But more efficient
             */
            const double val = 2.0 * matrixProduct;

            /** Go to next element */
            for (; (rowIt != rowVector.end()) && (rowIt->first < nmB); ++rowIt)
            {
            }

            if ((rowIt == rowVector.end()) || (rowIt->first != nmB))
            {
              /** Add new column to the row and set iterator to that column. */
              rowIt = rowVector.insert(rowIt, ElementType(nmB, val));
            }
            else
            {
              /** Add to existing value */
              rowIt->second += val;
            }
          }
        }
      }
    } // end if sampleOk
  }   // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Compute the measure value and derivative. */
  if (this->m_NumberOfPixelsCounted > 0)
  {
    const double normal_sum = 1.0 / static_cast<double>(this->m_NumberOfPixelsCounted);
    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      H.scale_row(i, normal_sum);
    }
  }
  else
  {
    // H.fill_diagonal(1.0);
    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      H(i, i) = 1.0;
    }
  }

} // end GetSelfHessian()


} // end namespace itk

#endif // #ifndef itkTransformBendingEnergyPenaltyTerm_hxx
