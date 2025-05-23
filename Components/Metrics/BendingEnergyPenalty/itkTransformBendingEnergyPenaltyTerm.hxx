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

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template <typename TFixedImage, typename TScalarType>
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::TransformBendingEnergyPenaltyTerm()
{
  /** Initialize member variables. */

  /** Turn on the sampler functionality. */
  this->SetUseImageSampler(true);

} // end Constructor


/**
 * ****************** GetValue *******************************
 */

template <typename TFixedImage, typename TScalarType>
auto
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetValue(const ParametersType & parameters) const
  -> MeasureType
{
  /** Initialize some variables. */
  Superclass::m_NumberOfPixelsCounted = 0;
  RealType           measure{};
  SpatialHessianType spatialHessian;

  /** Check if the SpatialHessian is nonzero. */
  if (!Superclass::m_AdvancedTransform->GetHasNonZeroSpatialHessian())
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

  /** Loop over the fixed image samples to calculate the penalty term. */
  for (const auto & fixedImageSample : *sampleContainer)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fixedImageSample.m_ImageCoordinates;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    if (sampleOk)
    {
      Superclass::m_NumberOfPixelsCounted++;

      /** Get the spatial Hessian of the transformation at the current point.
       * This is needed to compute the bending energy.
       */
      Superclass::m_AdvancedTransform->GetSpatialHessian(fixedPoint, spatialHessian);

      /** Compute the contribution of this point. */
      for (unsigned int k = 0; k < FixedImageDimension; ++k)
      {
        measure += vnl_math::sqr(spatialHessian[k].GetVnlMatrix().frobenius_norm());
      }

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples();

  /** Update measure value. */
  measure /= static_cast<RealType>(Superclass::m_NumberOfPixelsCounted);

  /** Return the value. */
  return static_cast<MeasureType>(measure);

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <typename TFixedImage, typename TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetDerivative(const ParametersType & parameters,
                                                                           DerivativeType &       derivative) const
{
  /** Slower, but works. */
  MeasureType dummyvalue{};
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ****************** GetValueAndDerivativeSingleThreaded *******************************
 */

template <typename TFixedImage, typename TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetValueAndDerivativeSingleThreaded(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  /** Create and initialize some variables. */
  Superclass::m_NumberOfPixelsCounted = 0;
  RealType measure{};
  derivative.set_size(this->GetNumberOfParameters());
  derivative.Fill(0.0);

  SpatialHessianType           spatialHessian;
  JacobianOfSpatialHessianType jacobianOfSpatialHessian;
  NonZeroJacobianIndicesType   nonZeroJacobianIndices;
  const NumberOfParametersType numberOfNonZeroJacobianIndices =
    Superclass::m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  jacobianOfSpatialHessian.resize(numberOfNonZeroJacobianIndices);
  nonZeroJacobianIndices.resize(numberOfNonZeroJacobianIndices);

  /** Check if the SpatialHessian is nonzero. */
  if (!Superclass::m_AdvancedTransform->GetHasNonZeroSpatialHessian() &&
      !Superclass::m_AdvancedTransform->GetHasNonZeroJacobianOfSpatialHessian())
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

  /** Loop over the fixed image to calculate the penalty term and its derivative. */
  for (const auto & fixedImageSample : *sampleContainer)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fixedImageSample.m_ImageCoordinates;

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
      Superclass::m_NumberOfPixelsCounted++;

      /** Get the spatial Hessian of the transformation at the current point.
       * This is needed to compute the bending energy.
       */
      //       Superclass::m_AdvancedTransform->GetSpatialHessian( fixedPoint,
      //         spatialHessian );
      //       Superclass::m_AdvancedTransform->GetJacobianOfSpatialHessian( fixedPoint,
      //         jacobianOfSpatialHessian, nonZeroJacobianIndices );
      Superclass::m_AdvancedTransform->GetJacobianOfSpatialHessian(
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
  this->CheckNumberOfSamples();

  /** Update measure value. */
  measure /= static_cast<RealType>(Superclass::m_NumberOfPixelsCounted);
  derivative /= static_cast<RealType>(Superclass::m_NumberOfPixelsCounted);

  /** The return value. */
  value = static_cast<MeasureType>(measure);

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <typename TFixedImage, typename TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetValueAndDerivative(const ParametersType & parameters,
                                                                                   MeasureType &          value,
                                                                                   DerivativeType & derivative) const
{
  /** Option for now to still use the single threaded code. */
  if (!Superclass::m_UseMultiThread)
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

template <typename TFixedImage, typename TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::ThreadedGetValueAndDerivative(ThreadIdType threadId) const
{
  /** Create and initialize some variables. */
  SpatialHessianType           spatialHessian;
  JacobianOfSpatialHessianType jacobianOfSpatialHessian;
  NonZeroJacobianIndicesType   nonZeroJacobianIndices;
  const NumberOfParametersType numberOfNonZeroJacobianIndices =
    Superclass::m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  jacobianOfSpatialHessian.resize(numberOfNonZeroJacobianIndices);
  nonZeroJacobianIndices.resize(numberOfNonZeroJacobianIndices);

  /** Check if the SpatialHessian is nonzero. */
  if (!Superclass::m_AdvancedTransform->GetHasNonZeroSpatialHessian() &&
      !Superclass::m_AdvancedTransform->GetHasNonZeroJacobianOfSpatialHessian())
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
  DerivativeType & derivative = Superclass::m_GetValueAndDerivativePerThreadVariables[threadId].st_Derivative;

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const size_t                sampleContainerSize{ sampleContainer->size() };

  /** Get the samples for this thread. */
  const auto nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  const auto pos_begin = std::min<size_t>(nrOfSamplesPerThreads * threadId, sampleContainerSize);
  const auto pos_end = std::min<size_t>(nrOfSamplesPerThreads * (threadId + 1), sampleContainerSize);

  /** Create iterator over the sample container. */
  const auto beginOfSampleContainer = sampleContainer->cbegin();
  const auto fbegin = beginOfSampleContainer + pos_begin;
  const auto fend = beginOfSampleContainer + pos_end;

  /** Create variables to store intermediate results. circumvent false sharing */
  unsigned long numberOfPixelsCounted = 0;
  MeasureType   measure{};

  /** Loop over the fixed image to calculate the penalty term and its derivative. */
  for (auto fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fiter->m_ImageCoordinates;

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
      Superclass::m_AdvancedTransform->GetJacobianOfSpatialHessian(
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
    } // end if sampleOk
  } // end for loop over the image sample container

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  Superclass::m_GetValueAndDerivativePerThreadVariables[threadId].st_NumberOfPixelsCounted = numberOfPixelsCounted;
  Superclass::m_GetValueAndDerivativePerThreadVariables[threadId].st_Value = measure;

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template <typename TFixedImage, typename TScalarType>
void
TransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::AfterThreadedGetValueAndDerivative(
  MeasureType &    value,
  DerivativeType & derivative) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Accumulate the number of pixels. */
  Superclass::m_NumberOfPixelsCounted = 0;
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    Superclass::m_NumberOfPixelsCounted +=
      Superclass::m_GetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples();

  /** Accumulate and normalize values. */
  value = MeasureType{};
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    value += Superclass::m_GetValueAndDerivativePerThreadVariables[i].st_Value;

    /** Reset this variable for the next iteration. */
    Superclass::m_GetValueAndDerivativePerThreadVariables[i].st_Value = MeasureType{};
  }
  value /= static_cast<RealType>(Superclass::m_NumberOfPixelsCounted);

  /** Accumulate derivatives. */
  // it seems that multi-threaded adding is faster than single-threaded
  // compute single-threadedly
  if (!Superclass::m_UseMultiThread)
  {
    derivative = Superclass::m_GetValueAndDerivativePerThreadVariables[0].st_Derivative;
    for (ThreadIdType i = 1; i < numberOfThreads; ++i)
    {
      derivative += Superclass::m_GetValueAndDerivativePerThreadVariables[i].st_Derivative;
    }
    derivative /= static_cast<DerivativeValueType>(Superclass::m_NumberOfPixelsCounted);
  }
  // compute multi-threadedly with itk threads
  else
  {
    Superclass::m_ThreaderMetricParameters.st_DerivativePointer = derivative.begin();
    Superclass::m_ThreaderMetricParameters.st_NormalizationFactor =
      static_cast<DerivativeValueType>(Superclass::m_NumberOfPixelsCounted);

    this->m_Threader->SetSingleMethodAndExecute(this->AccumulateDerivativesThreaderCallback,
                                                &(Superclass::m_ThreaderMetricParameters));
  }

} // end AfterThreadedGetValueAndDerivative()


} // end namespace itk

#endif // #ifndef itkTransformBendingEnergyPenaltyTerm_hxx
