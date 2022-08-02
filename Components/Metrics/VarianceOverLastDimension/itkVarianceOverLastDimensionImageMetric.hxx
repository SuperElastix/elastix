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
#ifndef itkVarianceOverLastDimensionImageMetric_hxx
#define itkVarianceOverLastDimensionImageMetric_hxx

#include "itkVarianceOverLastDimensionImageMetric.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include <vnl/algo/vnl_matrix_update.h>
#include <numeric>

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::VarianceOverLastDimensionImageMetric()
{
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);

} // end Constructor


/**
 * ******************* Initialize *******************
 */

template <class TFixedImage, class TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

  /** Check num last samples. */
  if (this->m_NumSamplesLastDimension > lastDimSize)
  {
    this->m_NumSamplesLastDimension = lastDimSize;
  }

  /** Compute variance over last dimension for complete image to use as normalization factor. */
  ImageLinearConstIteratorWithIndex<MovingImageType> it(this->GetMovingImage(),
                                                        this->GetMovingImage()->GetLargestPossibleRegion());
  it.SetDirection(lastDim);
  it.GoToBegin();

  float sumvar = 0.0;
  int   num = 0;
  while (!it.IsAtEnd())
  {
    /** Compute sum of values and sum of squared values. */
    float        sum = 0.0;
    float        sumsq = 0.0;
    unsigned int numlast = 0;
    while (!it.IsAtEndOfLine())
    {
      float value = it.Get();
      sum += value;
      sumsq += value * value;
      ++numlast;
      ++it;
    }

    /** Compute expected value (mean) and variance. */
    float expectedValue = sum / static_cast<float>(numlast);
    sumvar += sumsq / static_cast<float>(numlast) - expectedValue * expectedValue;
    ++num;

    it.NextLine();
  }

  /** Compute average variance. */
  if (sumvar == 0)
  {
    this->m_InitialVariance = 1.0f;
  }
  else
  {
    this->m_InitialVariance = sumvar / static_cast<float>(num);
  }

} // end Initialize()


/**
 * ******************* PrintSelf *******************
 */

template <class TFixedImage, class TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
} // end PrintSelf()


/**
 * ******************* SampleRandom *******************
 */

template <class TFixedImage, class TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::SampleRandom(const int          n,
                                                                              const int          m,
                                                                              std::vector<int> & numbers) const
{
  /** Empty list of last dimension positions. */
  numbers.clear();

  /** Initialize random number generator. */
  Statistics::MersenneTwisterRandomVariateGenerator::Pointer randomGenerator =
    Statistics::MersenneTwisterRandomVariateGenerator::GetInstance();

  /** Sample additional at fixed timepoint. */
  for (unsigned int i = 0; i < m_NumAdditionalSamplesFixed; ++i)
  {
    numbers.push_back(this->m_ReducedDimensionIndex);
  }

  /** Get n random samples. */
  for (int i = 0; i < n; ++i)
  {
    int randomNum = 0;
    do
    {
      randomNum = static_cast<int>(randomGenerator->GetVariateWithClosedRange(m));
    } while (find(numbers.begin(), numbers.end(), randomNum) != numbers.end());
    numbers.push_back(randomNum);
  }
} // end SampleRandom()


/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template <class TFixedImage, class TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::EvaluateTransformJacobianInnerProduct(
  const TransformJacobianType &     jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType &                  imageJacobian) const
{
  using JacobianIteratorType = typename TransformJacobianType::const_iterator;
  JacobianIteratorType jac = jacobian.begin();
  imageJacobian.Fill(0.0);

  for (unsigned int dim = 0; dim < FixedImageDimension; ++dim)
  {
    const double imDeriv = movingImageDerivative[dim];

    for (auto & imageJacobianElement : imageJacobian)
    {
      imageJacobianElement += (*jac) * imDeriv;
      ++jac;
    }
  }
} // end EvaluateTransformJacobianInnerProduct()


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
auto
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::GetValue(
  const TransformParametersType & parameters) const -> MeasureType
{
  itkDebugMacro("GetValue( " << parameters << " ) ");

  /** Initialize some variables */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;

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

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);
  const unsigned int numLastDimSamples = this->m_NumSamplesLastDimension;

  /** Vector containing last dimension positions to use:
   * initialize on all positions when random sampling turned off.
   */
  std::vector<int> lastDimPositions;
  if (!this->m_SampleLastDimensionRandomly)
  {
    for (unsigned int i = 0; i < lastDimSize; ++i)
    {
      lastDimPositions.push_back(i);
    }
  }

  /** Loop over the fixed image samples to calculate the variance over time for every sample position. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Determine random last dimension positions if needed. */
    if (this->m_SampleLastDimensionRandomly)
    {
      this->SampleRandom(numLastDimSamples, lastDimSize, lastDimPositions);
    }

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    /** Loop over the slowest varying dimension. */
    float              sumValues = 0.0;
    float              sumValuesSquared = 0.0;
    unsigned int       numSamplesOk = 0;
    const unsigned int realNumLastDimPositions = lastDimPositions.size();
    for (unsigned int d = 0; d < realNumLastDimPositions; ++d)
    {
      /** Initialize some variables. */
      RealType movingImageValue;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[lastDim] = lastDimPositions[d];

      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);

      /** Transform point. */
      const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

      /** Check if the point is inside the moving mask. */
      bool sampleOk = this->IsInsideMovingMask(mappedPoint);

      /** Compute the moving image value and check if the point is
       * inside the moving image buffer.
       */
      if (sampleOk)
      {
        sampleOk = this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, nullptr);
      }

      if (sampleOk)
      {
        ++numSamplesOk;
        sumValues += movingImageValue;
        sumValuesSquared += movingImageValue * movingImageValue;
      } // end if sampleOk
    }   // end for loop over last dimension

    if (numSamplesOk > 0)
    {
      this->m_NumberOfPixelsCounted++;

      /** Add this variance to the variance sum. */
      const float expectedValue = sumValues / static_cast<float>(numSamplesOk);
      const float expectedSquaredValue = sumValuesSquared / static_cast<float>(numSamplesOk);
      measure += expectedSquaredValue - expectedValue * expectedValue;
    }

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Compute average over variances. */
  measure /= static_cast<float>(this->m_NumberOfPixelsCounted);
  /** Normalize with initial variance. */
  measure /= this->m_InitialVariance;

  /** Return the mean squares measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::GetDerivative(
  const TransformParametersType & parameters,
  DerivativeType &                derivative) const
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
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
VarianceOverLastDimensionImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType &                   value,
  DerivativeType &                derivative) const
{
  itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

  /** Define derivative and Jacobian types. */
  using DerivativeValueType = typename DerivativeType::ValueType;

  /** Initialize some variables */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits<DerivativeValueType>::ZeroValue());

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

  /** Retrieve slowest varying dimension and its size. */
  const unsigned int lastDim = this->GetFixedImage()->GetImageDimension() - 1;
  const unsigned int lastDimSize = this->GetFixedImage()->GetLargestPossibleRegion().GetSize(lastDim);

  /** Vector containing last dimension positions to use:
   * initialize on all positions when random sampling turned off.
   */
  std::vector<int> lastDimPositions;
  if (!this->m_SampleLastDimensionRandomly)
  {
    for (unsigned int i = 0; i < lastDimSize; ++i)
    {
      lastDimPositions.push_back(i);
    }
  }

  /** Create variables to store intermediate results in. */
  TransformJacobianType jacobian;
  DerivativeType        imageJacobian(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());

  /** Get real last dim samples. */
  const unsigned int realNumLastDimPositions = this->m_SampleLastDimensionRandomly
                                                 ? this->m_NumSamplesLastDimension + this->m_NumAdditionalSamplesFixed
                                                 : lastDimSize;

  /** Variable to store and nzjis. */
  std::vector<NonZeroJacobianIndicesType> nzjis(realNumLastDimPositions, NonZeroJacobianIndicesType());

  std::vector<RealType>       MT(realNumLastDimPositions);
  std::vector<DerivativeType> dMTdmu(realNumLastDimPositions);

  /** Loop over the fixed image samples to calculate the variance over time for every sample position. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates. */
    FixedImagePointType fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Determine random last dimension positions if needed. */
    if (this->m_SampleLastDimensionRandomly)
    {
      this->SampleRandom(this->m_NumSamplesLastDimension, lastDimSize, lastDimPositions);
    }

    /** Initialize MT vector. */
    std::fill(MT.begin(), MT.end(), itk::NumericTraits<RealType>::ZeroValue());

    /** Transform sampled point to voxel coordinates. */
    auto voxelCoord =
      this->GetFixedImage()->template TransformPhysicalPointToContinuousIndex<CoordinateRepresentationType>(fixedPoint);

    /** Loop over the slowest varying dimension. */
    float        sumValues = 0.0;
    float        sumValuesSquared = 0.0;
    unsigned int numSamplesOk = 0;

    /** First loop over t: compute M(T(x,t)), dM(T(x,t))/dmu, nzji and store. */
    for (unsigned int d = 0; d < realNumLastDimPositions; ++d)
    {
      /** Initialize some variables. */
      RealType                  movingImageValue;
      MovingImageDerivativeType movingImageDerivative;

      /** Set fixed point's last dimension to lastDimPosition. */
      voxelCoord[lastDim] = lastDimPositions[d];
      /** Transform sampled point back to world coordinates. */
      this->GetFixedImage()->TransformContinuousIndexToPhysicalPoint(voxelCoord, fixedPoint);
      /** Transform point. */
      const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

      /** Check if the point is inside the moving mask. */
      bool sampleOk = this->IsInsideMovingMask(mappedPoint);

      /** Compute the moving image value and check if the point is
       * inside the moving image buffer. */
      if (sampleOk)
      {
        sampleOk = this->Superclass::EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, &movingImageDerivative);
      }

      if (sampleOk)
      {
        /** Update value terms **/
        ++numSamplesOk;
        sumValues += movingImageValue;
        sumValuesSquared += movingImageValue * movingImageValue;

        /** Get the TransformJacobian dT/dmu. */
        this->EvaluateTransformJacobian(fixedPoint, jacobian, nzjis[d]);

        /** Compute the innerproduct (dM/dx)^T (dT/dmu). */
        this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

        /** Store values. */
        MT[d] = movingImageValue;
        dMTdmu[d] = imageJacobian;
      }
      else
      {
        dMTdmu[d] = DerivativeType(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
        dMTdmu[d].Fill(itk::NumericTraits<DerivativeValueType>::ZeroValue());
        nzjis[d] = NonZeroJacobianIndicesType(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices(), 0);
      } // end if sampleOk
    }

    if (numSamplesOk > 0)
    {
      this->m_NumberOfPixelsCounted++;

      /** Compute average intensity value. */
      const float expectedValue = sumValues / static_cast<float>(numSamplesOk);
      /** Add this variance to the variance sum. */
      const float expectedSquaredValue = sumValuesSquared / static_cast<float>(numSamplesOk);
      measure += expectedSquaredValue - expectedValue * expectedValue;

      /** Second loop over t: update derivative. */
      for (unsigned int d = 0; d < realNumLastDimPositions; ++d)
      {
        for (unsigned int j = 0; j < nzjis[d].size(); ++j)
        {
          derivative[nzjis[d][j]] += (2.0 * (MT[d] - expectedValue) * dMTdmu[d][j]) / static_cast<float>(numSamplesOk);
        }
      }
    }
  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Compute average over variances and normalize with initial variance. */
  measure /= static_cast<float>(this->m_NumberOfPixelsCounted * this->m_InitialVariance);
  derivative /= static_cast<float>(this->m_NumberOfPixelsCounted * this->m_InitialVariance);

  /** Subtract mean from derivative elements. */
  if (this->m_SubtractMean)
  {
    if (!this->m_TransformIsStackTransform)
    {
      /** Update derivative per dimension.
       * Parameters are ordered xxxxxxx yyyyyyy zzzzzzz ttttttt and
       * per dimension xyz.
       */
      const unsigned int lastDimGridSize = this->m_GridSize[lastDim];
      const unsigned int numParametersPerDimension =
        this->GetNumberOfParameters() / this->GetMovingImage()->GetImageDimension();
      const unsigned int numControlPointsPerDimension = numParametersPerDimension / lastDimGridSize;
      DerivativeType     mean(numControlPointsPerDimension);
      for (unsigned int d = 0; d < this->GetMovingImage()->GetImageDimension(); ++d)
      {
        /** Compute mean per dimension. */
        mean.Fill(0.0);
        const unsigned int starti = numParametersPerDimension * d;
        for (unsigned int i = starti; i < starti + numParametersPerDimension; ++i)
        {
          const unsigned int index = i % numControlPointsPerDimension;
          mean[index] += derivative[i];
        }
        mean /= static_cast<double>(lastDimGridSize);

        /** Update derivative for every control point per dimension. */
        for (unsigned int i = starti; i < starti + numParametersPerDimension; ++i)
        {
          const unsigned int index = i % numControlPointsPerDimension;
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
      const unsigned int numParametersPerLastDimension = this->GetNumberOfParameters() / lastDimSize;
      DerivativeType     mean(numParametersPerLastDimension);
      mean.Fill(0.0);

      /** Compute mean per control point. */
      for (unsigned int t = 0; t < lastDimSize; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned int index = c % numParametersPerLastDimension;
          mean[index] += derivative[c];
        }
      }
      mean /= static_cast<double>(lastDimSize);

      /** Update derivative per control point. */
      for (unsigned int t = 0; t < lastDimSize; ++t)
      {
        const unsigned int startc = numParametersPerLastDimension * t;
        for (unsigned int c = startc; c < startc + numParametersPerLastDimension; ++c)
        {
          const unsigned int index = c % numParametersPerLastDimension;
          derivative[c] -= mean[index];
        }
      }
    }
  }

  /** Return the measure value. */
  value = measure;

} // end GetValueAndDerivative()


} // end namespace itk

#endif // end #ifndef _itkVarianceOverLastDimensionImageMetric_hxx
