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
#ifndef __itkStackTransformBendingEnergyPenaltyTerm_hxx
#define __itkStackTransformBendingEnergyPenaltyTerm_hxx

#include "itkStackTransformBendingEnergyPenaltyTerm.h"

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif

namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template <class TFixedImage, class TScalarType>
StackTransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::StackTransformBendingEnergyPenaltyTerm()
{

  this->SetUseImageSampler(true);

} // end Constructor


/**
 * ****************** GetValue *******************************
 */

template <class TFixedImage, class TScalarType>
typename StackTransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::MeasureType
StackTransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetValue(const ParametersType & parameters) const
{
  itkDebugMacro("GetValue( " << parameters << " ) ");

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

  /** Initialize some variables */
  this->m_NumberOfPixelsCounted = 0;
  RealType measure = NumericTraits<RealType>::Zero;

  /** Return 0 if the transform has zero spatial Hessian */
  SpatialHessianType spatialHessian;
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

  /** Loop over the fixed image samples to calculate the variance over time for every sample position. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates. */
    FixedImagePointType           fixedPoint = (*fiter).Value().m_ImageCoordinates;
    FixedImageContinuousIndexType voxelCoord;
    this->GetFixedImage()->TransformPhysicalPointToContinuousIndex(fixedPoint, voxelCoord);

    std::vector<FixedImagePointType> fixedPoints(lastDimSize);
    unsigned int                     numSamplesOk = 0;

    for (unsigned int s = 0; s < lastDimSize; s++)
    {
      RealType             movingImageValueTemp;
      MovingImagePointType mappedPoint;

      /** Set fixed point's last dimension to s. */
      voxelCoord[ReducedFixedImageDimension] = s;

      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);

      /** Transform point and check if it is inside the B-spline support region. */
      bool sampleOk = this->TransformPoint(fixedPoint, mappedPoint);

      /** Check if point is inside mask. */
      if (sampleOk)
      {
        sampleOk = this->IsInsideMovingMask(mappedPoint);
      }

      if (sampleOk)
      {
        fixedPoints[numSamplesOk] = fixedPoint;
        numSamplesOk++;
      }
    } // end for loop over last dimension

    if (numSamplesOk == lastDimSize)
    {
      this->m_NumberOfPixelsCounted++;
      // loop over last dimension
      for (unsigned int o = 0; o < lastDimSize; o++)
      {
        this->m_AdvancedTransform->GetSpatialHessian(fixedPoints[o], spatialHessian);

        // loop over spatial dimensions
        for (unsigned int k = 0; k < ReducedFixedImageDimension; k++)
        {
          measure += vnl_math_sqr(spatialHessian[k].GetVnlMatrix().frobenius_norm()) / numSamplesOk;
        } // end loop over spatial dimensions
      }   // end loop over last dimension
    }
  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Normalize with amount of pixels counted. */
  measure /= static_cast<RealType>(this->m_NumberOfPixelsCounted);

  /** Return the metric value. */
  return static_cast<MeasureType>(measure);

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedImage, class TScalarType>
void
StackTransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetDerivative(const ParametersType & parameters,
                                                                                DerivativeType &       derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits<MeasureType>::Zero;
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ****************** GetValueAndDerivativeSingleThreaded *******************************
 */

template <class TFixedImage, class TScalarType>
void
StackTransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetValueAndDerivativeSingleThreaded(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

  FixedImageSizeType m_GridSize;

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
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = (*fiter).Value().m_ImageCoordinates;

    /** Transform sampled point to voxel coordinates. */
    FixedImageContinuousIndexType voxelCoord;
    this->GetFixedImage()->TransformPhysicalPointToContinuousIndex(fixedPoint, voxelCoord);

    std::vector<FixedImagePointType> fixedPoints(lastDimSize);
    unsigned int                     numSamplesOk = 0;

    /** Loop over last dimension. */
    for (unsigned int s = 0; s < lastDimSize; s++)
    {
      /** Initialize some variables. */
      RealType             movingImageValueTemp;
      MovingImagePointType mappedPoint;

      /** Set fixed point's last dimension to s. */
      voxelCoord[ReducedFixedImageDimension] = s;
      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);
      /** Transform point and check if it is inside the B-spline support region. */
      bool sampleOk = this->TransformPoint(fixedPoint, mappedPoint);

      /** Check if point is inside mask. */
      if (sampleOk)
      {
        sampleOk = this->IsInsideMovingMask(mappedPoint);
      }

      if (sampleOk)
      {
        fixedPoints[s] = fixedPoint;
        numSamplesOk++;
      }
    } // end loop over last dimension
    /** Check if all points are valid. */
    if (numSamplesOk == lastDimSize)
    {
      this->m_NumberOfPixelsCounted++;
      /** loop over last dimension. */
      for (unsigned int o = 0; o < lastDimSize; o++)
      {
        /** Get the spatial Hessian of the transformation at the current point.
         * This is needed to compute the bending energy.
         */
        this->m_AdvancedTransform->GetJacobianOfSpatialHessian(
          fixedPoints[o], spatialHessian, jacobianOfSpatialHessian, nonZeroJacobianIndices);

        /** Prepare some stuff for the computation of the metric (derivative). */
        FixedArray<InternalMatrixType, ReducedFixedImageDimension> A;
        for (unsigned int k = 0; k < ReducedFixedImageDimension; k++)
        {
          A[k] = spatialHessian[k].GetVnlMatrix();
        }

        /** Compute the contribution to the metric value of this point. */
        for (unsigned int k = 0; k < ReducedFixedImageDimension; k++)
        {
          measure += vnl_math_sqr(spatialHessian[k].GetVnlMatrix().frobenius_norm()) / lastDimSize;
        }

        /** Double checking the transform is a stracktransform */
        if (!this->m_TransformIsBSpline)
        {
          /** Make a distinction between a B-spline subtransform and other subtransforms. */
          if (!this->m_SubTransformIsBSpline)
          {
            for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
            {
              for (unsigned int k = 0; k < ReducedFixedImageDimension; ++k)
              {
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

                derivative[nonZeroJacobianIndices[mu]] += 2.0 * matrixProduct / lastDimSize;
              }
            }
          }
          else
          {
            /** For the B-spline transform we know that only 1/FixedImageDimension
             * part of the JacobianOfSpatialHessian is non-zero.
             */
            /** Compute the contribution to the metric derivative of this point. */
            unsigned int numParPerDim = nonZeroJacobianIndices.size() / ReducedFixedImageDimension;

            for (unsigned int mu = 0; mu < numParPerDim; ++mu)
            {
              for (unsigned int k = 0; k < ReducedFixedImageDimension; ++k)
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

                derivative[nonZeroJacobianIndices[mu + numParPerDim * k]] += 2.0 * matrixElementProduct / lastDimSize;
              }
            }
          } // end if subtransform is B-spline
        }
        else
        {
          itkExceptionMacro(<< "This metric can only be used in combination with a StackTransform");
        } // end if subtransform is B-spline
      }   // end loop last dimension
    }     // end if samplesOk
  }       // end loop over sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Subtract mean from derivative elements. */
  if (this->m_SubtractMean)
  {
    if (!this->m_TransformIsStackTransform)
    {
      /** Update derivative per dimension.
       * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
       * per dimension xyz.
       */
      const unsigned int  lastDimGridSize = this->m_GridSize[lastDim];
      const unsigned long numParametersPerDimension =
        this->GetNumberOfParameters() / this->GetMovingImage()->GetImageDimension();
      const unsigned long numControlPointsPerDimension = numParametersPerDimension / lastDimGridSize;
      DerivativeType      mean(numControlPointsPerDimension);
      for (unsigned int d = 0; d < this->GetMovingImage()->GetImageDimension(); ++d)
      {
        /** Compute mean per dimension. */
        mean.Fill(0.0);
        const unsigned long starti = numParametersPerDimension * d;
        for (unsigned long i = starti; i < starti + numParametersPerDimension; ++i)
        {
          const unsigned long index = i % numControlPointsPerDimension;
          mean[index] += derivative[i];
        }
        mean /= static_cast<double>(lastDimGridSize);

        /** Update derivative for every control point per dimension. */
        for (unsigned long i = starti; i < starti + numParametersPerDimension; ++i)
        {
          const unsigned long index = i % numControlPointsPerDimension;
          derivative[i] -= mean[index];
        }
      }
    }
    else
    {
      /** Update derivative per dimension.
       * Parameters are ordered x0x0x0y0y0y0z0z0z0x1x1x1y1y1y1z1z1z1 with
       * the number the time point index.
       */
      const unsigned long numParametersPerLastDimension = this->GetNumberOfParameters() / lastDimSize;
      DerivativeType      mean(numParametersPerLastDimension);
      mean.Fill(0.0);

      /** Compute mean per control point. */
      for (unsigned int t = 0; t < lastDimSize; ++t)
      {
        const unsigned long startc = numParametersPerLastDimension * t;
        for (unsigned long c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned long index = c % numParametersPerLastDimension;
          mean[index] += derivative[c];
        }
      }
      mean /= static_cast<double>(lastDimSize);

      /** Update derivative per control point. */
      for (unsigned long t = 0; t < lastDimSize; ++t)
      {
        const unsigned long startc = numParametersPerLastDimension * t;
        for (unsigned long c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned long index = c % numParametersPerLastDimension;
          derivative[c] -= mean[index];
        }
      }
    }
  }

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
StackTransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
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
StackTransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::ThreadedGetValueAndDerivative(ThreadIdType threadId)
{
  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

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
    vcl_ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(this->m_NumberOfThreads)));

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
    /** Read fixed coordinates. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;

    /** Transform sampled point to voxel coordinates. */
    FixedImageContinuousIndexType voxelCoord;
    this->GetFixedImage()->TransformPhysicalPointToContinuousIndex(fixedPoint, voxelCoord);

    std::vector<FixedImagePointType> fixedPoints(lastDimSize);
    unsigned int                     numSamplesOk = 0;

    /** Loop over last dimension. */
    for (unsigned int s = 0; s < lastDimSize; s++)
    {
      /** Initialize some variables. */
      RealType             movingImageValueTemp;
      MovingImagePointType mappedPoint;
      FixedImagePointType  tempPoint;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[ReducedFixedImageDimension] = s;
      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, tempPoint);
      /** Transform point and check if it is inside the B-spline support region. */
      bool sampleOk = this->TransformPoint(tempPoint, mappedPoint);

      /** Check if point is inside mask. */
      if (sampleOk)
      {
        sampleOk = this->IsInsideMovingMask(mappedPoint);
      }
      if (sampleOk)
      {
        fixedPoints[s] = tempPoint;
        numSamplesOk++;
      }
    }

    if (numSamplesOk == lastDimSize)
    {
      numberOfPixelsCounted++;
      /** Loop over last dimension.*/
      for (unsigned int o = 0; o < lastDimSize; o++)
      {
        /** Get the spatial Hessian of the transformation at the current point.
         * This is needed to compute the bending energy.
         */
        this->m_AdvancedTransform->GetJacobianOfSpatialHessian(
          fixedPoints[o], spatialHessian, jacobianOfSpatialHessian, nonZeroJacobianIndices);

        /** Prepare some stuff for the computation of the metric (derivative). */
        FixedArray<InternalMatrixType, ReducedFixedImageDimension> A;
        for (unsigned int k = 0; k < ReducedFixedImageDimension; k++)
        {
          A[k] = spatialHessian[k].GetVnlMatrix();
        }

        /** Compute the contribution to the metric value of this point. */
        for (unsigned int k = 0; k < ReducedFixedImageDimension; k++)
        {
          measure += vnl_math_sqr(spatialHessian[k].GetVnlMatrix().frobenius_norm()) / numSamplesOk;
        }

        /** Double checking to make sure the transform is a stacktransform */
        if (!this->m_TransformIsBSpline)
        {
          /** Make a distinction between a B-spline subtransform and other subtransforms. */
          if (!this->m_SubTransformIsBSpline)
          {
            /** Compute the contribution to the metric derivative of this point. */
            for (unsigned int mu = 0; mu < nonZeroJacobianIndices.size(); ++mu)
            {
              for (unsigned int k = 0; k < ReducedFixedImageDimension; ++k)
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
                derivative[nonZeroJacobianIndices[mu]] += 2.0 * matrixProduct / numSamplesOk;
              }
            }
          }
          else
          {
            /** For the B-spline transform we know that only 1/FixedImageDimension
             * part of the JacobianOfSpatialHessian is non-zero.
             */

            /** Compute the contribution to the metric derivative of this point. */
            unsigned int numParPerDim = nonZeroJacobianIndices.size() / ReducedFixedImageDimension;
            for (unsigned int mu = 0; mu < numParPerDim; ++mu)
            {
              for (unsigned int k = 0; k < ReducedFixedImageDimension; ++k)
              {
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
                derivative[nonZeroJacobianIndices[mu + numParPerDim * k]] += 2.0 * matrixElementProduct / numSamplesOk;
              }
            }
          } // end if subtransform is B-spline
        }
        else
        {
          itkExceptionMacro(<< "This metric can only be used in combination with a StackTransform");
        } // end if transform is B-spline
      }   // end loop over last dim
    }     // end if sampleOk
  }       // end loop over sample container

  this->m_GetValueAndDerivativePerThreadVariables[threadId].st_NumberOfPixelsCounted = numberOfPixelsCounted;
  this->m_GetValueAndDerivativePerThreadVariables[threadId].st_Value = measure;

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TScalarType>
void
StackTransformBendingEnergyPenaltyTerm<TFixedImage, TScalarType>::AfterThreadedGetValueAndDerivative(
  MeasureType &    value,
  DerivativeType & derivative) const
{
  const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted = 0;
  for (ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i)
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
  for (ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i)
  {
    value += this->m_GetValueAndDerivativePerThreadVariables[i].st_Value;

    /** Reset this variable for the next iteration. IS THIS REALLY NECESSARY???*/
    this->m_GetValueAndDerivativePerThreadVariables[i].st_Value = NumericTraits<MeasureType>::Zero;
  }
  value /= static_cast<RealType>(this->m_NumberOfPixelsCounted);

  /** Accumulate derivatives. */
  // compute single-threadedly
  derivative = this->m_GetValueAndDerivativePerThreadVariables[0].st_Derivative;
  this->m_GetValueAndDerivativePerThreadVariables[0].st_Derivative.Fill(0.0);
  for (ThreadIdType i = 1; i < this->m_NumberOfThreads; ++i)
  {
    derivative += this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative;
    this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative.Fill(0.0);
  }
  derivative /= static_cast<DerivativeValueType>(this->m_NumberOfPixelsCounted);

  /** Subtract mean from derivative elements. */
  if (this->m_SubtractMean)
  {
    if (!this->m_TransformIsStackTransform)
    {
      /** Update derivative per dimension.
       * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
       * per dimension xyz.
       */
      const unsigned int  lastDimGridSize = this->m_GridSize[lastDim];
      const unsigned long numParametersPerDimension =
        this->GetNumberOfParameters() / this->GetMovingImage()->GetImageDimension();
      const unsigned long numControlPointsPerDimension = numParametersPerDimension / lastDimGridSize;
      DerivativeType      mean(numControlPointsPerDimension);
      for (unsigned int d = 0; d < this->GetMovingImage()->GetImageDimension(); ++d)
      {
        /** Compute mean per dimension. */
        mean.Fill(0.0);
        const unsigned long starti = numParametersPerDimension * d;
        for (unsigned long i = starti; i < starti + numParametersPerDimension; ++i)
        {
          const unsigned long index = i % numControlPointsPerDimension;
          mean[index] += derivative[i];
        }
        mean /= static_cast<double>(lastDimGridSize);

        /** Update derivative for every control point per dimension. */
        for (unsigned long i = starti; i < starti + numParametersPerDimension; ++i)
        {
          const unsigned long index = i % numControlPointsPerDimension;
          derivative[i] -= mean[index];
        }
      }
    }
    else
    {
      /** Update derivative per dimension.
       * Parameters are ordered x0x0x0y0y0y0z0z0z0x1x1x1y1y1y1z1z1z1 with
       * the number the time point index.
       */
      const unsigned long numParametersPerLastDimension = this->GetNumberOfParameters() / lastDimSize;
      DerivativeType      mean(numParametersPerLastDimension);
      mean.Fill(0.0);

      /** Compute mean per control point. */
      for (unsigned int t = 0; t < lastDimSize; ++t)
      {
        const unsigned long startc = numParametersPerLastDimension * t;
        for (unsigned long c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned long index = c % numParametersPerLastDimension;
          mean[index] += derivative[c];
        }
      }
      mean /= static_cast<double>(lastDimSize);

      /** Update derivative per control point. */
      for (unsigned long t = 0; t < lastDimSize; ++t)
      {
        const unsigned long startc = numParametersPerLastDimension * t;
        for (unsigned long c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned long index = c % numParametersPerLastDimension;
          derivative[c] -= mean[index];
        }
      }
    }
  }
} // end AfterThreadedGetValueAndDerivative()

} // end namespace itk

#endif // #ifndef __itkStackTransformBendingEnergyPenaltyTerm_hxx
